"""Owned worker for the separately frozen tiny device protocol."""
import argparse
from pathlib import Path
import time

from scoped_proof.io import Events, load, save, sha, tick
from scoped_proof import device_lifecycle as engine
from source_enclosure.format import identity
from scripts import hz_physical_device as policy
from scripts import hz_device_admission as admission
from scripts.hz_device_lifecycle import cleanup_confirmed


def fake_snapshot(gpu_uuid, *, busy):
    now=time.monotonic(); gpu=f'{gpu_uuid}, 97887, {32000 if busy else 34}, {95 if busy else 0}'
    processes=f'{gpu_uuid}, 99999999' if busy else ''
    return {**admission.parse(gpu,processes,gpu_uuid),'start':now,'end':time.monotonic(),
            'gpu_text':gpu,'process_text':processes}


def run(root,inv_sha,plan_sha):
    inv,p=engine.plan(policy,root,inv_sha,plan_sha); deadline=p['run_deadline']; tick(deadline)
    phase=p['phase']; inputs=p['inputs']; s=inv['spec']; control=s['control']; events=Events(root,phase,inv['start'])
    def mark():
        save(root/(phase+'_fault.json'),{'control':control,'invocation':inv['invocation'],'physical_cuda':False})
        events.emit('FAULT',operation=control,physical_cuda=False)
    def delay():
        mark(); time.sleep(max(1.,inv['deadline']-time.monotonic()+1))
    if phase=='admit':
        if inputs: raise ValueError('admission inputs')
        if control in ('admission_busy_stub','pre_cuda_busy_stub'):
            observations=[fake_snapshot(s['gpu_uuid'],busy=control=='admission_busy_stub')]
            if control=='pre_cuda_busy_stub':
                time.sleep(.2); observations.append(fake_snapshot(s['gpu_uuid'],busy=False))
            else: mark()
            result={'invocation':inv['invocation'],'device':s['device'],'gpu_uuid':s['gpu_uuid'],
                'deadline':deadline,'exclusive_reservation':False,'simulated':True,'observations':observations,
                'status':'RESOURCE_UNAVAILABLE' if control=='admission_busy_stub' else 'READY'}
        else:
            result=events.call('admission',lambda:admission.observe(s,inv,deadline))
    elif phase=='produce':
        stage=load(root/'admit_stage.json')
        if inputs!={'admit':stage['output_sha256']} or not policy.admitted(root,inv,stage): raise ValueError('producer admission')
        import torch
        torch.set_num_threads(1); original_init=torch.cuda._lazy_init
        def forbidden(*args,**kwargs): raise AssertionError('CUDA not yet admitted / CPU-only worker')
        torch.cuda._lazy_init=forbidden
        from scripts.hz_batch_support_worker import fixture
        from act.back_end.moe.batched_support import prepare_batch
        from act.back_end.moe import batched_support_device as device
        hz,queries,context=events.call('fresh_HZ_creation',lambda:fixture(s['case']))
        batch=events.call('prepare',lambda:prepare_batch(hz,queries,context=context,deadline=deadline))
        if identity(batch)!=s['batch_sha256']: raise ValueError('frozen HZ identity')
        events.call('batch_publication',lambda:save(root/'batch.json',batch))
        if control=='producer_exception': mark(); raise RuntimeError('controlled producer exception')
        pre_sha=None; ready=True
        if s['device']=='cuda:0':
            if control=='pre_cuda_busy_stub': mark()
            observation=events.call('pre_cuda_recheck',lambda:fake_snapshot(s['gpu_uuid'],busy=True)
                if control=='pre_cuda_busy_stub' else admission.snapshot(s['gpu_uuid'],deadline))
            pre={'invocation':inv['invocation'],'simulated':bool(control),'ready':observation['ready'],'observation':observation}
            pre_sha=save(root/'pre_cuda.json',pre)['sha256']; policy.validate_pre(root,inv,pre); ready=pre['ready']
        result={'status':'RESOURCE_UNAVAILABLE','invocation':inv['invocation'],'spec_sha256':identity(s),
                'pre_cuda_sha256':pre_sha,'cuda_intent_sha256':None}
        if ready:
            if s['device']=='cuda:0':
                if control: raise ValueError('no control may initialize CUDA')
                torch.cuda._lazy_init=original_init
                result['cuda_intent_sha256']=save(root/'cuda_intent.json',{'invocation':inv['invocation'],'pre_cuda_sha256':pre_sha,
                                             'producer_pid':__import__('os').getpid(),'not_completion_evidence':True})['sha256']
            ctx=policy.context(inv,deadline)
            candidates=events.call('candidate_API',lambda:device.propose_batch(batch,expected_batch_sha256=s['batch_sha256'],
                deadline=deadline,device=s['device'],cuda_context=ctx,expected_context_sha256=None if ctx is None else identity(ctx)))
            result.update(status='CANDIDATES',batch=batch,candidates=candidates,context=ctx,
                          environment={'torch':torch.__version__,'cuda_build':torch.version.cuda,'cpu_threads':torch.get_num_threads()})
    elif phase=='release':
        producer=load(root/'produce_stage.json',inputs['produce'],engine.LIMIT)
        if not cleanup_confirmed(producer): raise ValueError('cleanup required before release')
        if control=='release_delay': delay()
        if s['device']=='cpu':
            result={'status':'CPU_NO_CUDA_TO_RELEASE','invocation':inv['invocation'],'producer_sha256':inputs['produce']}
        else:
            rows=[]
            for i in range(2):
                if i: time.sleep(.05)
                rows.append(fake_snapshot(s['gpu_uuid'],busy=False) if control else admission.snapshot(s['gpu_uuid'],deadline))
            result={'schema':'OWNED_DEVICE_RELEASE_OBSERVATION_V1','invocation':inv['invocation'],
                'gpu_uuid':s['gpu_uuid'],'producer_sha256':inputs['produce'],'owned_pids':[producer['pid']],
                'after':inv['start']+producer['end_seconds'],'deadline':deadline,'simulated':bool(control),'observations':rows}
    elif phase in ('check','receive'):
        producer=load(root/'produce_stage.json'); observer=load(root/'release_stage.json')
        if (inputs['produce']!=producer['output_sha256'] or inputs['release']!=observer['output_sha256']
                or not policy.release_gate(root,inv,{'produce':producer,'release':observer})['release_confirmed']):
            raise ValueError('parent candidate/release anchor')
        if phase=='check':
            if control=='check_delay': delay()
            result=events.call('exact_check',lambda:policy.check(root,inv,inputs['produce'],deadline))
        else: result=events.call('reception',lambda:policy.receive(root,inv,inputs))
    else: raise ValueError('phase')
    tick(deadline); events.call('publication',lambda:save(root/(phase+'.json'),result)); tick(deadline)


def preflight(root,deadline,invocation,gpu_uuid):
    spec={'device':'cuda:0','gpu_uuid':gpu_uuid,'control':''}
    result=admission.observe(spec,{'invocation':invocation},deadline)
    tick(deadline); save(root/'probe.json',result); tick(deadline)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    p.add_argument('--invocation-sha'); p.add_argument('--plan-sha'); p.add_argument('--preflight',action='store_true')
    p.add_argument('--deadline',type=float); p.add_argument('--invocation'); p.add_argument('--gpu-uuid')
    a=p.parse_args()
    if a.preflight: preflight(a.root,a.deadline,a.invocation,a.gpu_uuid)
    else: run(a.root,a.invocation_sha,a.plan_sha)
