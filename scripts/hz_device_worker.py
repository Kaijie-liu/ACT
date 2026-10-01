"""Owned device-support workers. Physical CUDA needs separately frozen execution."""
import argparse
from pathlib import Path
import subprocess
import time

from scoped_proof.io import PYTHON, Events, load, save, tick
from source_enclosure.format import identity
from scripts.hz_device_supervised import bind, required_hash, validate_spec, receive, LIMIT, phase_deadline, bound_admission, metadata
from scripts import hz_device_admission as admission


from scripts.hz_batch_support_worker import fixture


def run(phase, root, deadline, invocation_sha, payload_sha=None, stdout_sha=None, admission_sha=None):
    spec = load(root/'spec.json')
    validate_spec(spec)
    inv = bind(root, spec, invocation_sha)
    expected_end = phase_deadline(inv,phase)
    if deadline != expected_end: raise ValueError('worker deadline extension/change')
    tick(deadline)
    events = Events(root, phase, inv['start'])
    fault = spec['control']
    def delay(name):
        save(root/(name+'_unaccepted.json'), {'control':name,'invocation':inv['invocation'],'accepted':False})
        time.sleep(10)
    if fault == phase+'_delay': delay(phase)
    if phase=='admit':
        if fault=='admission_error_stub':
            events.emit('FAULT',operation=fault)
            raise RuntimeError('simulated admission query failure; no CUDA')
        record=events.call('bounded_resource_observation',lambda: admission.observe(spec,inv,deadline))
        events.call('admission_publication',lambda: save(root/'admission.json',record))
    elif phase == 'produce':
        gate=bound_admission(root,spec,inv,admission_sha)
        if gate['status'] not in ('CPU_NO_CUDA','READY'): raise ValueError('resource not admitted')
        if fault == 'descendant':
            child = subprocess.Popen([PYTHON,'-B','-c','import time; time.sleep(10)'])
            save(root/'descendant_control.json',{'pid':child.pid,'invocation':inv['invocation']})
            return
        hz, queries, context = events.call('imports_and_hz_creation', lambda: fixture(spec['case']))
        import torch
        torch.set_num_threads(1)
        from act.back_end.moe.batched_support import prepare_batch
        from act.back_end.moe import batched_support_device as device
        # CPU controls explicitly forbid accidental lazy CUDA initialization.
        if spec['device']=='cpu':
            def forbidden_cuda(*args,**kwargs): raise AssertionError('unexpected CUDA in CPU worker')
            torch.cuda._lazy_init=forbidden_cuda
        pre_cuda_sha=None
        if spec['device']=='cuda:0':
            fresh=events.call('pre_cuda_resource_recheck',lambda: admission.snapshot(spec['gpu_uuid'],deadline))
            pre_cuda_sha=save(root/'pre_cuda_observation.json',fresh)['sha256']
            if not fresh['ready']: raise RuntimeError('resource changed before CUDA; no initialization')
        def prepare():
            if fault == 'prepare_delay': delay('prepare')
            return prepare_batch(hz,queries,context=context,deadline=deadline)
        batch = events.call('hz_export_and_validation', prepare)
        if identity(batch) != spec['batch_sha256']: raise ValueError('created HZ/request differs from frozen case')
        events.call('batch_publication', lambda: save(root/'batch.json', batch))
        if fault == 'candidate_exception':
            events.emit('FAULT',operation=fault)
            raise RuntimeError('controlled candidate exception')
        def propose():
            from unittest.mock import patch
            original_sync,original_readback=device._synchronize,device._readback
            count=[0]
            def sync(which):
                count[0]+=1
                if fault=='device_sync_stub' and count[0]==3:
                    events.emit('FAULT',operation=fault,physical_cuda=False)
                    raise RuntimeError('simulated synchronize failure')
                return original_sync(which)
            def readback(*args):
                value=original_readback(*args)
                if fault=='readback_delay_stub':
                    events.emit('FAULT',operation=fault,physical_cuda=False)
                    delay('readback')
                return value
            if fault=='device_oom_stub':
                events.emit('FAULT',operation=fault,physical_cuda=False)
                raise torch.OutOfMemoryError('simulated allocator failure')
            context=inv['cuda_context']
            with patch.object(device,'_synchronize',side_effect=sync),patch.object(device,'_readback',side_effect=readback):
                return device.propose_batch(batch,expected_batch_sha256=spec['batch_sha256'],deadline=deadline,
                    device=spec['device'],cuda_context=context,
                    expected_context_sha256=None if context is None else identity(context))
        candidates=events.call('device_candidates_and_exact_evaluation',propose)
        if fault=='wrong_device_metadata': candidates['device']='cuda:0'
        if fault == 'partial_candidate': candidates['entries'].pop()
        if fault == 'wrong_query': batch['queries'][0]['side'] = 'max'
        payload = {'invocation':inv['invocation'], 'request_sha256':identity(spec),
                   'batch_sha256':spec['batch_sha256'], 'batch':batch, 'candidates':candidates,
                   'admission_sha256':admission_sha,'pre_cuda_sha256':pre_cuda_sha}
        if fault == 'wrong_invocation': payload['invocation'] = 'different-request'
        if fault == 'serialization_delay': delay('serialization')
        events.call('candidate_serialization', lambda: save(root/'payload.json',payload))
    elif phase == 'check':
        if fault == 'check_exception':
            events.emit('FAULT',operation=fault)
            raise RuntimeError('controlled check exception')
        if fault == 'missing_stdout':
            save(root/'missing_stdout_unaccepted.json',{'invocation':inv['invocation'],'control':fault,'accepted':False})
            return
        def check():
            payload = load(root/'payload.json', required_hash(payload_sha), limit=LIMIT)
            if (payload['invocation'] != inv['invocation'] or payload['request_sha256'] != identity(spec)
                    or payload['batch_sha256'] != spec['batch_sha256']):
                raise ValueError('candidate invocation/request binding')
            metadata(root,spec,inv,payload,admission_sha)
            from act.back_end.moe.check_batched_support import check_batch
            result = check_batch(payload['batch'],payload['candidates'],expected_batch_sha256=spec['batch_sha256'],deadline=deadline)
            return {'schema':'HZ_DEVICE_CHECK_OUTPUT_V1','invocation':inv['invocation'],
                    'request_sha256':identity(spec),'batch_sha256':spec['batch_sha256'],
                    'payload_sha256':payload_sha,'result':result}
        result = events.call('independent_exact_reception', check)
        events.call('checker_output_publication', lambda: save(root/'check.stdout',result))
    elif phase == 'receive':
        if fault == 'rewrite_stdout':
            # Controlled mutation of this invocation only; parent already anchored bytes.
            value = load(root/'check.stdout')
            value['result']['results'][0]['bound'] = '999'
            import json
            (root/'check.stdout').write_text(json.dumps(value))
        result = events.call('identity_and_full_roster_reception', lambda: receive(
            root,spec,invocation_sha,payload_sha,stdout_sha,deadline,admission_sha))
        if fault == 'late_publish':
            save(root/'late_candidate_unaccepted.json',result)
            delay('late_publish')
        events.call('accepted_publication', lambda: save(root/'accepted.json',result))
    else:
        raise ValueError('worker phase')
    tick(deadline)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase'); p.add_argument('root',type=Path)
    p.add_argument('--deadline',type=float,required=True); p.add_argument('--invocation-sha',required=True)
    p.add_argument('--payload-sha'); p.add_argument('--stdout-sha'); p.add_argument('--admission-sha')
    a=p.parse_args()
    run(a.phase,a.root,a.deadline,a.invocation_sha,a.payload_sha,a.stdout_sha,a.admission_sha)
