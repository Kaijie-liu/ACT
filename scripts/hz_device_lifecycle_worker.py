"""Thin CPU-only dispatcher under immutable parent launch plans."""
import argparse
from pathlib import Path
import time

from scoped_proof.io import Events, load, save, tick
from source_enclosure.format import identity
from scripts import hz_device_lifecycle as lifecycle
from scripts.hz_device_release import simulated_record


def run(root, invocation_sha, plan_sha):
    inv,plan=lifecycle.validate_plan(root,invocation_sha,plan_sha)
    phase=plan['phase']; deadline=plan['run_deadline']; inputs=plan['inputs']
    tick(deadline)
    events=Events(root,phase,inv['start']); s=inv['spec']; fault=s['control']

    def mark():
        events.emit('FAULT',operation=fault,physical_cuda=False)
        save(root/(phase+'_fault.json'),{'invocation':inv['invocation'],'control':fault,'accepted':False})

    def delay():
        mark(); time.sleep(max(1.,inv['deadline']-time.monotonic()+1))

    if phase=='admit':
        if inputs: raise ValueError('unexpected admission inputs')
        result={'status':'CPU_ADMITTED_NO_CUDA','invocation':inv['invocation']}
    elif phase=='produce':
        if set(inputs)!={'admit'}: raise ValueError('producer inputs')
        if load(root/'admit.json',inputs['admit'],lifecycle.LIMIT)!={
                'status':'CPU_ADMITTED_NO_CUDA','invocation':inv['invocation']}:
            raise ValueError('producer admission binding')
        from scripts.hz_batch_support_worker import fixture
        hz,queries,context=events.call('imports_and_hz_creation',lambda:fixture(s['case']))
        import torch
        torch.set_num_threads(1)
        def forbidden_cuda(*args,**kwargs): raise AssertionError('physical CUDA not admitted')
        torch.cuda._lazy_init=forbidden_cuda
        from act.back_end.moe.batched_support import prepare_batch
        from act.back_end.moe import batched_support_device as device
        batch=events.call('hz_export_and_validation',lambda:prepare_batch(hz,queries,context=context,deadline=deadline))
        if identity(batch)!=s['batch_sha256']: raise ValueError('fixed batch identity')
        events.call('batch_prefix_publication',lambda:save(root/'batch.json',batch))
        if fault=='producer_exception':
            mark(); raise RuntimeError('controlled producer exception')
        def propose():
            from unittest.mock import patch
            original=device._readback
            def readback(*args):
                result=original(*args)
                if fault=='readback_stall': delay()
                return result
            with patch.object(device,'_readback',side_effect=readback):
                return device.propose_batch(batch,expected_batch_sha256=s['batch_sha256'],deadline=deadline,device='cpu')
        candidates=events.call('cpu_candidates_and_exact_evaluation',propose)
        result={'invocation':inv['invocation'],'spec_sha256':identity(s),'batch':batch,'candidates':candidates}
    elif phase=='release':
        if set(inputs)!={'produce'}: raise ValueError('release inputs')
        producer=load(root/'produce_stage.json',inputs['produce'],lifecycle.LIMIT)
        if producer['phase']!='produce' or not lifecycle.cleanup_confirmed(producer):
            raise ValueError('producer cleanup not established')
        if fault=='release_probe_error':
            mark(); raise RuntimeError('simulated release query error')
        if fault=='release_deadline': delay()
        if s['release_mode']=='simulated':
            mark()
            result=events.call('simulated_release_observations',lambda:simulated_record(inv,plan,producer,fault))
        else:
            result={'status':'CPU_NO_CUDA_TO_RELEASE','invocation':inv['invocation'],'producer_sha256':inputs['produce']}
    elif phase in ('check','receive'):
        expected={'produce','release'} | ({'check'} if phase=='receive' else set())
        if set(inputs)!=expected: raise ValueError('checker/receiver inputs')
        # The observer's parent-anchored bytes and the producer's cleanup remain
        # obligations, even though this worker never initializes CUDA.
        producer=load(root/'produce_stage.json',limit=lifecycle.LIMIT)
        observer=load(root/'release_stage.json',limit=lifecycle.LIMIT)
        if observer['output_sha256']!=inputs['release'] or producer['output_sha256']!=inputs['produce']:
            raise ValueError('parent release/candidate anchors')
        if not lifecycle.release_gate(root,inv,{'produce':producer,'release':observer}).get('release_confirmed'):
            raise ValueError('resource release not established')
        if phase=='check':
            result=events.call('independent_exact_check',lambda:lifecycle.mathematical_check(root,inv,inputs['produce'],deadline))
        else:
            result=events.call('complete_roster_reception',lambda:lifecycle.receive(root,inv,inputs))
    else: raise ValueError('phase')
    tick(deadline)
    events.call('publication',lambda:save(root/(phase+'.json'),result))
    tick(deadline)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path); p.add_argument('--invocation-sha',required=True); p.add_argument('--plan-sha',required=True)
    a=p.parse_args(); run(a.root,a.invocation_sha,a.plan_sha)
