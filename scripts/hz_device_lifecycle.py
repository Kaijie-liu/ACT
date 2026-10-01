"""Versioned CPU/stub lifecycle coordinator; original numerical functions reused.

Physical CUDA is deliberately not admitted by this frozen protocol. Process
cleanup and release observation are separate obligations, including failed work.
"""
from fractions import Fraction
import os
from pathlib import Path
import time
import uuid

from scoped_proof.io import ROOT, PYTHON, load, save, sha, tick
from scoped_proof.owned_bounded import execute
from source_enclosure.format import identity
from scripts.hz_device_supervised import FILES as DEVICE_FILES
from scripts.hz_propagation_supervised import finite, required_hash
from scripts import hz_device_release as release

CONFIG = 'configs/hz_device_lifecycle_20261001.json'
PROTOCOL_SHA = 'e79af4a14a42d769f81ec2a89bb56cad9be6b3237147d64a8119a310848b0d98'
DONE = 'CHECKED_GIVEN_HZ_CPU_DEVICE_LIFECYCLE'
LIMIT = 4*2**20
PHASES = ('admit','produce','release','check','receive')
PENDING = ('CLEANUP_INCOMPLETE','DEVICE_RELEASE_UNCONFIRMED')
FILES = tuple(dict.fromkeys([*DEVICE_FILES, CONFIG, 'scoped_proof/owned_bounded.py',
    'scripts/hz_propagation_supervised.py', 'scripts/hz_device_lifecycle.py',
    'scripts/hz_device_release.py', 'scripts/hz_device_lifecycle_worker.py']))


def protocol():
    cfg = load(ROOT/CONFIG, PROTOCOL_SHA)
    for name,digest in cfg['frozen_dependencies'].items():
        if sha(ROOT/name) != digest: raise ValueError('frozen dependency changed: '+name)
    old = load(ROOT/cfg['reference_protocol'], cfg['reference_protocol_sha256'])
    for name,digest in old['frozen_dependencies'].items():
        if sha(ROOT/name)!=digest: raise ValueError('frozen numerical dependency changed: '+name)
    return cfg, old


def spec(case='guarded_two_sides_positive', control=''):
    cfg,old = protocol()
    if case not in cfg['cases'] or control not in ('',*cfg['faults']) or control and case!=cfg['cases'][0]:
        raise ValueError('fixed CPU control roster')
    simulated = control.startswith('release_')
    return {'schema':'HZ_DEVICE_LIFECYCLE_SPEC_V1', 'case':case, 'control':control,
            'device':'cpu','release_mode':'simulated' if simulated else 'cpu',
            'gpu_uuid':cfg['gpu_uuid_for_simulation'] if simulated else None,
            'protocol_sha256':PROTOCOL_SHA, **old['cases'][case]}


def sources():
    return {name:sha(ROOT/name) for name in FILES}


def cleanup_confirmed(stage):
    """Apply to failures too. No-process and reaped-process are distinct."""
    if stage.get('cleanup_unconfirmed_stub'): return False
    if stage.get('remaining_group',{}).get('live')!=[]: return False
    pid=stage.get('pid'); rc=stage.get('returncode')
    if pid is None: return stage.get('cleanup_status')=='NO_PROCESS' and rc is None
    return (type(pid) is int and pid>0 and type(rc) is int
            and stage.get('cleanup_status')=='LEADER_REAPED_NO_LIVE_GROUP')


def bind(root, digest):
    inv=load(root/'invocation.json',required_hash(digest),LIMIT)
    s=inv['spec']
    if s != spec(s['case'],s['control']) or inv['sources']!=sources():
        raise ValueError('source/request/device binding')
    for key in ('start','deadline','work','budget'): finite(inv[key])
    if (not 0<inv['budget']<=300 or inv['deadline']!=inv['start']+inv['budget']
            or inv['work']!=inv['deadline']-min(1.,inv['budget']/5)
            or inv['rss_limit']!=2*2**30): raise ValueError('budget contract')
    return inv


def cutoff(inv, phase, begin):
    if phase=='admit': return min(inv['work'],inv['start']+2)
    if phase=='produce': return inv['work']-min(6.,inv['budget']/2)
    if phase=='release': return min(inv['work'],begin+min(2.,inv['budget']/4))
    return inv['work']


def validate_plan(root, inv_sha, plan_sha):
    inv=bind(root,inv_sha)
    # Plans are immutable and named by phase; the worker is given the hash.
    matches=[]
    for phase in PHASES:
        path=root/(phase+'_plan.json')
        if path.is_file() and sha(path)==required_hash(plan_sha): matches.append(load(path,plan_sha,LIMIT))
    if len(matches)!=1: raise ValueError('unique parent-owned launch plan')
    plan=matches[0]; phase=plan['phase']; finite(plan['begin'])
    if (phase not in PHASES or plan['invocation_sha256']!=inv_sha or plan['begin']<inv['start']
            or plan['cleanup_deadline']!=cutoff(inv,phase,plan['begin'])
            or plan['run_deadline']!=plan['cleanup_deadline']-min(.25,inv['budget']/20)):
        raise ValueError('phase deadline/identity')
    return inv,plan


def payload(root, inv, digest):
    p=load(root/'produce.json',required_hash(digest),LIMIT); s=inv['spec']; c=p['candidates']
    if (p['invocation']!=inv['invocation'] or p['spec_sha256']!=identity(s)
            or identity(p['batch'])!=s['batch_sha256'] or c['batch_sha256']!=s['batch_sha256']
            or c['device']!='cpu' or c['dtype']!='float64' or c['iterations']!=128
            or c['algorithm']!='projected_dual_subgradient_multiobjective_v1'
            or c['execution_context_sha256'] is not None or c['hardware'] is not None
            or c['allocator_memory'] is not None or c['hard_budget_supervision'] is not False):
        raise ValueError('candidate/source/device contract')
    costs=c['cost_seconds']
    expected={'validation','host_tensors','device_initialization_sync','transfer_and_setup_sync',
              'optimization_sync','readback_sync','exact_candidate_evaluation','other','total'}
    if set(costs)!=expected: raise ValueError('candidate cost fields')
    for v in costs.values(): finite(v)
    if abs(costs['total']-sum(v for k,v in costs.items() if k!='total'))>1e-8:
        raise ValueError('nested candidate cost')
    if c.get('cost_scope')!='proposal API only; caller must charge HZ preparation, independent checking, publication and cleanup':
        raise ValueError('proposal cost scope')
    producer=load(root/'produce_stage.json',limit=LIMIT)
    finite(producer['seconds'])
    if producer.get('output_sha256')!=digest or costs['total']>producer['seconds']:
        raise ValueError('parent candidate anchor/nested cost exceeds producer')
    return p


def mathematical_check(root, inv, digest, deadline):
    p=payload(root,inv,digest)
    from act.back_end.moe.check_batched_support import check_batch
    checked=check_batch(p['batch'],p['candidates'],expected_batch_sha256=inv['spec']['batch_sha256'],deadline=deadline)
    return {'invocation':inv['invocation'],'payload_sha256':digest,'result':checked}


def receive(root, inv, inputs):
    p=payload(root,inv,inputs['produce'])
    c=load(root/'check.json',required_hash(inputs['check']),LIMIT); r=c['result']; s=inv['spec']
    if (c['invocation']!=inv['invocation'] or c['payload_sha256']!=inputs['produce']
            or r['status']!='CHECKED_GIVEN_HZ_CONTINUOUS_RELAXATION'
            or r['batch_sha256']!=s['batch_sha256'] or r['candidate_sha256']!=identity(p['candidates'])
            or [row['id'] for row in r['results']]!=s['query_ids']
            or [row['side'] for row in r['results']]!=s['sides']
            or r['network_or_complete_moe_proof'] is not False or r['hard_budget_supervision'] is not False):
        raise ValueError('full exact check reception')
    for row in r['results']:
        Fraction(row['bound']); required_hash(row['lp_sha256'])
        if row['bound_kind'] != ('lower' if row['side']=='min' else 'upper'):
            raise ValueError('bound direction')
    return {'status':DONE,'invocation':inv['invocation'],'spec_sha256':identity(s),
            'inputs':inputs,'checked':r,'complete_moe_proof':False,'physical_cuda':False}


def release_gate(root, inv, records):
    producer=records['produce']; observed=records['release']
    if not cleanup_confirmed(producer) or not cleanup_confirmed(observed):
        return {'status':'DEVICE_RELEASE_UNCONFIRMED','reason':'cleanup not confirmed'}
    if observed['status']!='COMPLETED':
        return {'status':'DEVICE_RELEASE_UNCONFIRMED','reason':'release phase incomplete'}
    try:
        r=load(root/'release.json',required_hash(observed['output_sha256']),LIMIT)
        producer_sha=sha(root/'produce_stage.json')
        if inv['spec']['release_mode']=='cpu':
            if r!={'status':'CPU_NO_CUDA_TO_RELEASE','invocation':inv['invocation'],'producer_sha256':producer_sha}:
                raise ValueError('CPU release binding')
            return {'status':'CPU_NO_CUDA_TO_RELEASE','release_confirmed':True,'next_admission_ready':None,
                    'physical_driver_release_proved':False}
        plan=load(root/'release_plan.json',observed['plan_sha256'])
        result=release.validate(r,invocation=inv['invocation'],gpu_uuid=inv['spec']['gpu_uuid'],
            producer_sha=producer_sha,producer_pid=producer['pid'],after=inv['start']+producer['end_seconds'],
            deadline=plan['run_deadline'],simulated=True)
        if not result['release_confirmed']: result['status']='DEVICE_RELEASE_UNCONFIRMED'
        return result
    except (ValueError,KeyError,TypeError) as exc:
        return {'status':'DEVICE_RELEASE_UNCONFIRMED','reason':str(exc)}


def decision(root, inv, stages):
    records={r['phase']:r for r in stages}
    if not stages: return 'ERROR',None,None
    for r in stages:
        if not cleanup_confirmed(r):
            return 'CLEANUP_INCOMPLETE',None,None
    if records['admit']['status']!='COMPLETED': return records['admit']['status'],None,None
    gate=load(root/'admit.json',required_hash(records['admit']['output_sha256']),LIMIT)
    if gate!={'status':'CPU_ADMITTED_NO_CUDA','invocation':inv['invocation']}:
        raise ValueError('CPU admission')
    if 'produce' not in records: return 'ERROR',None,None
    if 'release' not in records: return 'DEVICE_RELEASE_UNCONFIRMED',None,None
    release_result=release_gate(root,inv,records)
    if release_result['status']=='DEVICE_RELEASE_UNCONFIRMED': return 'DEVICE_RELEASE_UNCONFIRMED',release_result,None
    if records['produce']['status']!='COMPLETED': return records['produce']['status'],release_result,None
    for phase in ('check','receive'):
        if phase not in records: return 'ERROR',release_result,None
        if records[phase]['status']!='COMPLETED': return records[phase]['status'],release_result,None
    inputs={phase:records[phase]['output_sha256'] for phase in ('produce','release','check')}
    accepted=receive(root,inv,inputs)
    if load(root/'receive.json',records['receive']['output_sha256'],LIMIT)!=accepted:
        raise ValueError('receiver output mismatch')
    return DONE,release_result,accepted


def finalize_status(status, now, deadline):
    return status if status in PENDING else 'TIMEOUT' if now>=deadline else status


def supervise(root, request, *, request_sha256, budget=300.):
    start=time.monotonic(); finite(budget)
    if not 0<budget<=300 or request!=spec(request['case'],request['control']) or identity(request)!=required_hash(request_sha256):
        raise ValueError('frozen request and budget')
    root=Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project archive required')
    root.mkdir(parents=True,exist_ok=False)
    inv={'spec':request,'invocation':uuid.uuid4().hex,'start':start,'deadline':start+budget,
         'work':start+budget-min(1.,budget/5),'budget':budget,'rss_limit':2*2**30,'sources':sources()}
    inv_sha=save(root/'invocation.json',inv)['sha256']; stages=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')

    def call(phase, inputs):
        begin=time.monotonic(); end=cutoff(inv,phase,begin)
        plan={'phase':phase,'invocation_sha256':inv_sha,'begin':begin,'cleanup_deadline':end,
              'run_deadline':end-min(.25,budget/20),'inputs':inputs}
        plan_sha=save(root/(phase+'_plan.json'),plan)['sha256']
        command=[PYTHON,'-B']+(['-S'] if phase in ('admit','release','receive') else [])+[
            '-m','scripts.hz_device_lifecycle_worker',str(root),'--invocation-sha',inv_sha,'--plan-sha',plan_sha]
        r=execute(command,root/(phase+'.log'),run_deadline=plan['run_deadline'],cleanup_deadline=end,env=env,rss_limit=inv['rss_limit'])
        r.update(phase=phase,plan_sha256=plan_sha,start_seconds=begin-start,executable=PYTHON,
                 cpu_threads=1,cuda_visible_devices='')
        if r['status']=='COMPLETED':
            try:
                load(root/(phase+'.json'),limit=LIMIT)
                r['output_sha256']=sha(root/(phase+'.json'))
            except Exception as exc:
                r['status']='ERROR'; r['output_error']=repr(exc)
        if (phase=='produce' and request['control']=='cleanup_unconfirmed'
                and r['status']=='COMPLETED' and r['returncode']==0
                and r['cleanup_status']=='LEADER_REAPED_NO_LIVE_GROUP'
                and not r['remaining_group']['live'] and 'output_sha256' in r):
            r['cleanup_unconfirmed_stub']=True  # actual cleanup record is retained unchanged
        r['end_seconds']=time.monotonic()-start
        if r['status']=='COMPLETED' and start+r['end_seconds']>=end: r['status']='TIMEOUT'
        stages.append(r); save(root/(phase+'_stage.json'),r)
        return r

    status,error,rel,accepted='ERROR',None,None,None
    try:
        a=call('admit',{})
        if a['status']=='COMPLETED' and cleanup_confirmed(a):
            p=call('produce',{'admit':a['output_sha256']})
            if cleanup_confirmed(p):
                r=call('release',{'produce':sha(root/'produce_stage.json')})
                rel=release_gate(root,inv,{x['phase']:x for x in stages})
                if p['status']=='COMPLETED' and rel.get('release_confirmed'):
                    c=call('check',{'produce':p['output_sha256'],'release':r['output_sha256']})
                    if c['status']=='COMPLETED' and cleanup_confirmed(c):
                        call('receive',{'produce':p['output_sha256'],'release':r['output_sha256'],'check':c['output_sha256']})
        status,rel,accepted=decision(root,inv,stages)
    except Exception as exc:
        error=repr(exc)
        # A reception error must not hide an unconfirmed lifecycle obligation.
        if any(not cleanup_confirmed(s) for s in stages):
            status='CLEANUP_INCOMPLETE'
        elif any(s['phase']=='produce' for s in stages) and not (rel or {}).get('release_confirmed'):
            status='DEVICE_RELEASE_UNCONFIRMED'
    inventory={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    now=time.monotonic(); status=finalize_status(status,now,inv['work'])
    terminal={'invocation_sha256':inv_sha,'status':status,'error':error,'release':rel,
        'accepted':accepted if status==DONE else None,'stages':stages,'seconds':now-start,
        'stage_seconds':sum(r['seconds'] for r in stages),
        'inventory':inventory}
    terminal['parent_seconds']=terminal['seconds']-terminal['stage_seconds']
    terminal_sha=save(root/'terminal.json',terminal)['sha256']
    now=time.monotonic(); status=finalize_status(status,now,inv['deadline'])
    finish_sha=save(root/'finish.json',{'terminal_sha256':terminal_sha,'status':status,'seconds':now-start})['sha256']
    now=time.monotonic(); status=finalize_status(status,now,inv['deadline'])
    return {'status':status,'invocation_sha256':inv_sha,'finish_sha256':finish_sha,'seconds':now-start,
            'complete_moe_proof':False,'physical_cuda':False}
