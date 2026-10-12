"""Store/block device policy for the existing owned-process lifecycle."""
from pathlib import Path

from scoped_proof.io import ROOT, PYTHON, load, sha
from scoped_source.hz_request_store import identity, require
from scripts import hz_store_supervised as base
from scripts import hz_device_admission as admission
from scripts import hz_device_release as release

DONE='CHECKED_STORE_DEVICE_EXECUTION'
WORKER='scripts.hz_store_device_worker'
DESIGN='docs/hz_store_device_supervision_design_20261013.md'
UUID='GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc'
LIMIT=base.LIMIT
FAULTS=('admission_busy_stub','pre_cuda_busy_stub','candidate_exception','readback_delay',
        'release_owned_stays','release_delay','wrong_candidate_metadata')
GPU_STUBS=('admission_busy_stub','pre_cuda_busy_stub','release_owned_stays','release_delay')
EXTRA=('status','context','pre_cuda_sha256','cuda_intent_sha256','environment')
cleanup_confirmed=base.cleanup_confirmed
evidence_inventory=base.evidence_inventory
finite=base.base.finite
required_hash=base.base.required_hash


def sources():
    names=(*base.sources(),DESIGN,'scripts/hz_store_device_supervised.py',
           'scripts/hz_store_device_worker.py','scripts/test_hz_store_device_supervised.py',
           'scripts/run_hz_store_device_supervision.py','scripts/hz_device_admission.py',
           'scripts/hz_device_release.py','scripts/hz_physical_worker.py','scripts/hz_physical_device.py')
    return {n:sha(ROOT/n) for n in dict.fromkeys(names)}


def spec(case='half',device='cpu',control=''):
    require(case in ('half','negative','partial') and device in ('cpu','cuda:0')
            and control in ('',*FAULTS) and (not control or case=='half')
            and (not control or (device=='cuda:0')==(control in GPU_STUBS)), 'store device roster')
    s=base.spec(case)
    s.update(schema='STORE_DEVICE_EXECUTION_V1',control=control,device=device,
             gpu_uuid=UUID if device=='cuda:0' else None,design_sha256=sha(ROOT/DESIGN),
             algorithm=('CPU' if device=='cpu' else 'CUDA')+'_BLOCK_PROJECTED_SUBGRADIENT_128_V1')
    return s


def validate_spec(s,permit):
    require(s==spec(s['case'],s['device'],s['control']), 'frozen store/device spec')
    if permit is None:
        require(s['device']=='cpu' or s['control'] in GPU_STUBS, 'physical execution freeze required')
        return
    require(not s['control'] and type(permit) is dict and set(permit)=={'path','sha256'}, 'physical permit fields')
    path=Path(permit['path'])
    require(path.is_absolute() and path.resolve().parent==ROOT/'configs', 'committed execution freeze location')
    f=load(path,required_hash(permit['sha256']),LIMIT)
    require(f['schema']=='STORE_DEVICE_PHYSICAL_FREEZE_V1' and f['sources']==sources()
            and f['design_sha256']==sha(ROOT/DESIGN) and identity(s) in f['spec_sha256_roster'],
            'physical source/roster freeze')
    receipt=load(f['controls_audit'],required_hash(f['controls_audit_sha256']),LIMIT)
    require(receipt['status']=='PASS' and receipt['calls']==10 and receipt['cuda_calls']==0
            and receipt['sources']==sources() and receipt['tests']==12, 'matching lifecycle controls required')


def budget(s): return 300


def cutoff(inv,phase,begin):
    end=(min(inv['work'],begin+5) if phase in ('admit','release') else
         inv['work']-30 if phase=='produce' else inv['work']-3 if phase=='check' else inv['work'])
    c=inv['spec']['control']
    if c=='readback_delay' and phase=='produce': end=min(end,begin+15)
    return end


def visible(inv,phase):
    s=inv['spec']
    return s['gpu_uuid'] if phase=='produce' and s['device']=='cuda:0' and not s['control'] else ''


def worker_command(phase,root,inv_sha,plan_sha):
    args=[str(root),'--invocation-sha',inv_sha,'--plan-sha',plan_sha]
    if phase=='produce': return [PYTHON,'-B','-m',WORKER,*args]
    code=f'import sys,runpy;sys.path.insert(0,{str(ROOT)!r});runpy.run_module({WORKER!r},run_name="__main__")'
    return [PYTHON,'-B','-I','-S','-c',code,*args]


def context(inv,deadline,contract,anchor):
    s=inv['spec']
    if s['device']=='cpu': return None
    return dict(schema='STORE_DEVICE_CONTEXT_V1',store=contract['store'],request=s['source'],
                contract=anchor,deadline=deadline,invocation=inv['invocation'],
                gpu_uuid=s['gpu_uuid'],allocator_limit_bytes=2**30)


def admitted(root,inv,stage):
    r=load(root/'admit.json',required_hash(stage['output_sha256']),LIMIT)
    p=load(root/'admit_plan.json',stage['plan_sha256']); s=dict(inv['spec'])
    simulated=s['control'] in GPU_STUBS
    # The observer's original validator reserves one flag for simulated refusal.
    s['control']='admission_busy_stub' if simulated and r['status']=='RESOURCE_UNAVAILABLE' else ''
    c=dict(r,simulated=bool(s['control']))
    admission.validate(c,inv['invocation'],s,p['run_deadline'])
    require(r['simulated'] is simulated and all(o['start']>=p['begin'] for o in r['observations']),
            'fresh actual/simulated admission identity')
    return r['status'] in ('CPU_NO_CUDA','READY')


def validate_pre(root,inv,record):
    p=load(root/'produce_plan.json'); row=record['observation']; s=inv['spec']
    require(record['invocation']==inv['invocation'] and record['simulated'] is bool(s['control'])
            and record['ready']==row['ready'], 'pre-CUDA identity')
    rebuilt=admission.parse(row['gpu_text'],row['process_text'],s['gpu_uuid'])
    require(all(row.get(k)==v for k,v in rebuilt.items()), 'pre-CUDA raw observation')
    finite(row['start']); finite(row['end'])
    require(p['begin']<=row['start']<=row['end']<p['run_deadline'], 'pre-CUDA clock')


def producer_status(root,inv,digest):
    r=load(root/'produce.json',required_hash(digest),LIMIT)
    require(r['invocation']==inv['invocation'] and r['spec_sha256']==identity(inv['spec']), 'producer identity')
    if r['status']=='RESOURCE_UNAVAILABLE':
        require(inv['spec']['device']=='cuda:0' and r['cuda_intent_sha256'] is None
                and not (root/'cuda_intent.json').exists(), 'pre-CUDA refusal cannot initialize')
        pre=load(root/'pre_cuda.json',required_hash(r['pre_cuda_sha256']),LIMIT)
        validate_pre(root,inv,pre); require(not pre['ready'],'refusal without busy observation')
        return False
    require(r['status']=='CANDIDATES','producer status'); return True


def device_metadata(p,s,ctx,stage):
    c=p['candidate']; dev=c['device_report']
    require(c['device']==s['device'] and c['algorithm']==s['algorithm'] and c['dtype']=='float64'
            and c['iterations_per_tile']==128 and c['proposal_tiles']==3
            and c['proposal_columns']==(6 if s['case']=='negative' else 5), 'candidate algorithm/device')
    require(p['context']==ctx and dev['context_sha256']==(None if ctx is None else identity(ctx)), 'full device context')
    env=p['environment']
    require(set(env)=={'torch','cuda_build','cpu_threads'} and type(env['torch']) is str and env['torch']
            and type(env['cpu_threads']) is int and env['cpu_threads']==1, 'producer environment')
    if s['device']=='cpu':
        require(dev['hardware'] is None and dev['memory'] is None
                and p['pre_cuda_sha256'] is None and p['cuda_intent_sha256'] is None, 'CPU device metadata')
    else:
        h,m=dev['hardware'],dev['memory']
        require(h['gpu_uuid']==s['gpu_uuid'] and type(h['name']) is str and h['name']
                and type(env['cuda_build']) is str and env['cuda_build'], 'actual CUDA identity')
        finite(h['total_memory'],minimum=2**30)
        require(all(type(m[k]) is int for k in ('peak_allocated','peak_reserved'))
                and 0<=m['peak_allocated']<=m['peak_reserved']<=2**30,'allocator envelope')
    finite(c['api_seconds']); require(c['api_seconds']<=p['costs']['candidates']<=stage['seconds'],'candidate inclusive cost')
    cpu_phases=('native_preparation_seconds','proposal_seconds','exact_selection_seconds','serialization_seconds')
    for name in cpu_phases: finite(c[name])
    require(sum(c[name] for name in cpu_phases)<=c['api_seconds'],'candidate CPU phase envelope')
    require(set(dev['completed_device_costs'])=={'initialization_sync_seconds','transfer_sync_seconds',
            'optimization_sync_seconds','readback_sync_seconds'},'device cost inventory')
    for value in dev['completed_device_costs'].values():
        finite(value); require(value<=c['api_seconds'],'nested device cost')
    require(sum(dev['completed_device_costs'].values())<=c['api_seconds']
            and dev['transfer_groups']==10 and dev['cost_scope']==
            'nested in inclusive API/preparation/proposal costs; do not add twice', 'device cost/transfer scope')


def payload(root,inv,digest):
    p=base.source_payload(root,inv,digest,EXTRA); s=inv['spec']
    require(p['status']=='CANDIDATES','complete candidate status')
    stage=load(root/'produce_stage.json'); launch=load(root/'produce_plan.json',stage['plan_sha256'])
    contract=load(root/'evidence/contract.json',p['contract'],8192)
    ctx=context(inv,launch['run_deadline'],contract,p['contract']); device_metadata(p,s,ctx,stage)
    if s['device']=='cpu': require(not (root/'cuda_intent.json').exists(),'CPU CUDA intent')
    else:
        pre=load(root/'pre_cuda.json',required_hash(p['pre_cuda_sha256']),LIMIT); validate_pre(root,inv,pre)
        require(pre['ready'] and not pre['simulated'] and not s['control'],'real pre-CUDA admission')
        intent=load(root/'cuda_intent.json',required_hash(p['cuda_intent_sha256']),LIMIT)
        require(intent==dict(invocation=inv['invocation'],pre_cuda_sha256=p['pre_cuda_sha256'],
                producer_pid=stage['pid'],not_completion_evidence=True),'CUDA intent binding')
    return p


def produced(root,inv,stage):
    if not producer_status(root,inv,stage['output_sha256']): return False
    payload(root,inv,stage['output_sha256']); return True


def check(root,inv,digest,deadline):
    return base.check_payload(root,inv,digest,deadline,payload(root,inv,digest))


def receive(root,inv,inputs):
    result=base.receive_payload(root,inv,inputs,payload(root,inv,inputs['produce']))
    result.update(status=DONE,cuda_execution=inv['spec']['device']=='cuda:0')
    return result


def release_gate(root,inv,records):
    p,r=records['produce'],records['release']
    if not cleanup_confirmed(p) or not cleanup_confirmed(r) or r['status']!='COMPLETED':
        return dict(status='DEVICE_RELEASE_UNCONFIRMED',release_confirmed=False)
    value=load(root/'release.json',required_hash(r['output_sha256']),LIMIT); psha=sha(root/'produce_stage.json')
    if inv['spec']['device']=='cpu':
        require(value==dict(status='CPU_NO_CUDA_TO_RELEASE',invocation=inv['invocation'],producer_sha256=psha),
                'CPU release binding')
        return dict(status='CPU_NO_CUDA_TO_RELEASE',release_confirmed=True,next_admission_established=False)
    plan=load(root/'release_plan.json',r['plan_sha256'])
    result=release.validate(value,invocation=inv['invocation'],gpu_uuid=inv['spec']['gpu_uuid'],
        producer_sha=psha,producer_pid=p['pid'],after=inv['start']+p['end_seconds'],
        deadline=plan['run_deadline'],simulated=bool(inv['spec']['control']))
    if not result['release_confirmed']: result['status']='DEVICE_RELEASE_UNCONFIRMED'
    return result


def decision(root,inv,stages):
    r={v['phase']:v for v in stages}
    if any(not cleanup_confirmed(v) for v in stages): return 'CLEANUP_INCOMPLETE',None,None
    if r['admit']['status']!='COMPLETED': return r['admit']['status'],None,None
    if not admitted(root,inv,r['admit']): return 'RESOURCE_UNAVAILABLE',None,None
    if 'produce' not in r: return 'ERROR',None,None
    if 'release' not in r: return 'DEVICE_RELEASE_UNCONFIRMED',None,None
    rel=release_gate(root,inv,r)
    if not rel['release_confirmed']: return 'DEVICE_RELEASE_UNCONFIRMED',rel,None
    if r['produce']['status']!='COMPLETED': return r['produce']['status'],rel,None
    if not producer_status(root,inv,r['produce']['output_sha256']): return 'RESOURCE_UNAVAILABLE',rel,None
    # A rejected producer envelope may have completed before rejection; audit
    # its ERROR prefix without treating it as a successful payload.
    for phase in ('check','receive'):
        if phase not in r: return 'ERROR',rel,None
        if r[phase]['status']!='COMPLETED': return r[phase]['status'],rel,None
    result=receive(root,inv,{k:r[k]['output_sha256'] for k in ('produce','release','check')})
    require(load(root/'receive.json',r['receive']['output_sha256'],LIMIT)==result,'receiver bytes')
    return DONE,rel,result
