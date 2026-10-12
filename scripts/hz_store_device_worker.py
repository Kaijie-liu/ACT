"""Owned store/device worker; numerical imports occur only in production."""
import argparse
import os
from pathlib import Path
import time

from scoped_proof.io import Events, load, save, sha, tick
from scoped_proof import device_lifecycle as engine
from scoped_source.hz_request_store import Store, identity, require
from scoped_source.check_store_endpoints import EndpointWork, plan, SCHEMA
from scripts import hz_store_device_supervised as policy
from scripts import hz_device_admission as admission
from scripts import hz_device_release as release
from scripts.hz_physical_worker import fake_snapshot


def run(root,inv_sha,plan_sha):
    inv,p=engine.plan(policy,root,inv_sha,plan_sha)
    phase=p['phase']; end=p['run_deadline']; s=inv['spec']; control=s['control']
    events=Events(root,phase,inv['start']); tick(end)

    def fault(name):
        if control!=name: return
        events.emit('FAULT_REACHED',control=name,invocation=inv['invocation'],physical_cuda=False)
        if name.endswith('_delay'):
            while time.monotonic()<end+1: time.sleep(.05)
            tick(end)
        if name.endswith('_exception'): raise RuntimeError('registered '+name)

    if phase=='admit':
        if control in policy.GPU_STUBS:
            busy=control=='admission_busy_stub'
            if busy: fault(control)
            obs=[fake_snapshot(s['gpu_uuid'],busy=busy)]
            if not busy:
                time.sleep(.2); obs.append(fake_snapshot(s['gpu_uuid'],busy=False))
            result=dict(invocation=inv['invocation'],device=s['device'],gpu_uuid=s['gpu_uuid'],
                deadline=end,exclusive_reservation=False,simulated=True,observations=obs,
                status='RESOURCE_UNAVAILABLE' if busy else 'READY')
        else: result=events.call('admission',lambda:admission.observe(s,inv,end))
    elif phase=='release':
        producer=load(root/'produce_stage.json',p['inputs']['produce'])
        require(policy.cleanup_confirmed(producer),'producer cleanup before release')
        if control in ('release_delay','release_owned_stays'): fault(control)
        if s['device']=='cpu':
            result=dict(status='CPU_NO_CUDA_TO_RELEASE',invocation=inv['invocation'],producer_sha256=p['inputs']['produce'])
        elif control:
            result=events.call('release_observation',lambda:release.simulated_record(inv,p,producer,control))
        else:
            def observe():
                rows=[admission.snapshot(s['gpu_uuid'],end)]; time.sleep(.05)
                rows.append(admission.snapshot(s['gpu_uuid'],end))
                return dict(schema='OWNED_DEVICE_RELEASE_OBSERVATION_V1',invocation=inv['invocation'],
                    gpu_uuid=s['gpu_uuid'],producer_sha256=p['inputs']['produce'],owned_pids=[producer['pid']],
                    after=inv['start']+producer['end_seconds'],deadline=end,simulated=False,observations=rows)
            result=events.call('release_observation',observe)
    elif phase=='produce':
        stage=load(root/'admit_stage.json')
        require(p['inputs']=={'admit':stage['output_sha256']} and policy.admitted(root,inv,stage),'producer admission')
        costs={}
        def timed(name,fn):
            before=time.monotonic(); value=events.call(name,fn); costs[name]=time.monotonic()-before; return value
        def imports():
            import torch
            torch.set_num_threads(1); original=torch.cuda._lazy_init
            def forbidden(*a,**k): raise AssertionError('CUDA not admitted / CPU-only control')
            torch.cuda._lazy_init=forbidden
            from scoped_source.hz_store_source import build
            from act.back_end.moe.store_endpoints import produce
            return torch,original,build,produce
        torch,original_init,build,produce=timed('imports',imports)
        work=EndpointWork(end); data=root/'evidence'; data.mkdir(exist_ok=False)
        declaration=policy.base.fixture(s['case']); require(identity(declaration)==s['source'],'frozen source')
        save(data/'source.json',declaration)
        store_sha=timed('source',lambda:build(data/'store',declaration,work=work)[0])
        def prepare():
            store=Store(data/'store',expected_manifest=store_sha,expected_request=s['source'],work=work)
            contract=dict(schema=SCHEMA,store=store_sha,request=s['source'],
                          **{k:s[k] for k in ('mode','classes','label','margin','semantics','ties')})
            anchor=save(data/'contract.json',contract)['sha256']
            inventory=plan(store,contract,expected_contract=anchor,declaration=declaration,work=work)
            events.emit('SOURCE_READY',source=s['source'],store=store_sha,contract=anchor,
                        counts=work.counts,required_endpoints=len(inventory.tasks))
            return store,contract,inventory,anchor
        store,contract,inventory,anchor=timed('plan',prepare)
        result=dict(status='RESOURCE_UNAVAILABLE',invocation=inv['invocation'],spec_sha256=identity(s),
                    pre_cuda_sha256=None,cuda_intent_sha256=None)
        def publish_event(name,i,ref):
            if name=='tile': events.emit('TILE_PUBLISHED',index=i,reference=ref,counts=work.counts,
                                        contract=anchor,required_endpoints=len(inventory.tasks))
        def candidates():
            if s['device']=='cuda:0':
                if control=='pre_cuda_busy_stub': fault(control)
                row=events.call('pre_cuda_recheck',lambda:fake_snapshot(s['gpu_uuid'],busy=True)
                    if control else admission.snapshot(s['gpu_uuid'],end))
                pre=dict(invocation=inv['invocation'],simulated=bool(control),ready=row['ready'],observation=row)
                result['pre_cuda_sha256']=save(root/'pre_cuda.json',pre)['sha256']
                policy.validate_pre(root,inv,pre)
                if not pre['ready']: return None
                require(not control,'simulated admission cannot initialize CUDA')
                result['cuda_intent_sha256']=save(root/'cuda_intent.json',dict(invocation=inv['invocation'],
                    pre_cuda_sha256=result['pre_cuda_sha256'],producer_pid=os.getpid(),not_completion_evidence=True))['sha256']
                torch.cuda._lazy_init=original_init
            ctx=policy.context(inv,end,contract,anchor)
            from act.back_end.moe import store_device as runtime
            # Faults exercise the actual new runtime, never a fake positive result.
            original_move=runtime.Runtime.move; original_read=runtime._readback; reads=[]
            def move(self,values):
                fault('candidate_exception'); return original_move(self,values)
            def read(*args):
                values=original_read(*args); reads.append(1)
                if len(reads)==5: fault('readback_delay')
                return values
            runtime.Runtime.move=move; runtime._readback=read
            try:
                proof,stats=produce(data/'proof',store,inventory,tile_width=s['tile_width'],work=work,
                    missing_slots=[inventory.tasks[-1].slot] if s['case']=='partial' else [],event=publish_event,
                    device=s['device'],cuda_context=ctx,expected_context_sha256=None if ctx is None else identity(ctx))
            finally: runtime.Runtime.move=original_move; runtime._readback=original_read
            if control=='wrong_candidate_metadata': fault(control); stats['device']='cuda:0'
            return dict(proof=proof,candidate=stats,context=ctx,contract=anchor,counts=work.counts,
                        environment=dict(torch=torch.__version__,cuda_build=torch.version.cuda,cpu_threads=torch.get_num_threads()))
        completed=timed('candidates',candidates)
        if completed is not None: result.update(status='CANDIDATES',costs=costs,**completed)
    else:
        stages={k:load(root/(k+'_stage.json')) for k in ('produce','release')}
        require(all(p['inputs'][k]==stages[k]['output_sha256'] for k in stages)
                and policy.release_gate(root,inv,stages)['release_confirmed'],'candidate/release parent anchors')
        result=events.call('independent_exact_check' if phase=='check' else 'reception',
            lambda:policy.check(root,inv,p['inputs']['produce'],end) if phase=='check' else policy.receive(root,inv,p['inputs']))
    tick(end); events.call('final_serialization',lambda:save(root/(phase+'.json'),result))
    tick(end); events.emit('WORKER_COMPLETE',phase=phase)


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('root',type=Path)
    a.add_argument('--invocation-sha',required=True); a.add_argument('--plan-sha',required=True); args=a.parse_args()
    run(args.root,args.invocation_sha,args.plan_sha)
