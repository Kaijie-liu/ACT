"""Owned synthetic worker; fault controls never select real models or inputs."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from scoped_proof.io import ROOT, PYTHON, Events, load, save, sha, tick
from source_enclosure.format import identity
from scoped_source.sparse_supervised import validate_spec, receive, bind_checker, bind_producer


def run(phase, root, deadline, invocation_sha):
    inv = load(root/'invocation.json',invocation_sha); spec = load(root/'spec.json'); validate_spec(spec)
    if identity(spec) != inv['spec_sha256'] or deadline > inv['work_deadline']:
        raise ValueError('invocation/deadline identity')
    bind_producer(spec,inv)
    events = Events(root,phase,inv['start']); fault = spec['control']; tick(deadline)
    if fault == phase+'_delay':
        save(root/(phase+'_partial.json'), {'state':'CONTROL_DELAY_NO_ACCEPTANCE'})
        time.sleep(10)
    if phase == 'produce':
        if fault == 'descendant':
            subprocess.Popen([PYTHON,'-B','-c','import time; time.sleep(10)'])
            return
        if fault == 'memory':
            large = bytearray(128*2**20); time.sleep(10); return
        if spec['schema'] == 'H1_CAPTURE_CONTROL_V1':
            from scoped_source.sparse_intake import model_fixture, capture_bound
            model, center, request = events.call('model_creation_imports',lambda:model_fixture(spec['fixture']))
            if fault in ('capture_delay','capture_exception'):
                save(root/'capture_partial.json',{'status':'UNACCEPTED_MODEL_CREATED'})
                if fault == 'capture_delay': time.sleep(10)
                else: raise RuntimeError('controlled capture exception')
            if fault == 'mutate_model':
                import torch
                with torch.no_grad(): next(model.parameters()).add_(1)
            doc = events.call('capture_and_validate',lambda:capture_bound(model,center,request,
                expected_source_sha256=spec['source_sha256'],deadline=deadline))
            save(root/'captured_source_identity.json',{'source_sha256':identity(doc),
                'model_state':doc['request']['model_state'],'request':doc['request'],
                'invocation':inv['invocation'],'producer_sources':inv['producer_sources'],
                'native_float_proof':False,'real_requests_started':0})
        else:
            from scoped_source.sparse_controls import source
            doc = events.call('source_creation',lambda:source(**spec['fixture']))
        if identity(doc) != spec['source_sha256']: raise ValueError('frozen synthetic source')
        def importer():
            from act.back_end.solver.lp_certificate import propose
            return propose
        proposer = events.call('proposal_imports',importer)
        from scoped_source.sparse_build import build
        reuse = [(tuple(p), j) for p,j in spec['reuse']]
        proof = events.call('construct_and_propose',lambda:build(doc,expected_source_sha256=spec['source_sha256'],
            deadline=deadline,mode=spec['mode'],reuse_keys=reuse,proposer=proposer))
        if fault == 'missing_certificate': proof['obligations'][0]['certificate'] = None
        if fault == 'omit_property': proof['obligations'].pop()
        from scoped_source.sparse_portable import pack
        built = events.call('serialize_and_bundle',lambda:pack(root/'bundle',doc,proof,
            spec['source_sha256'],inv['invocation'],deadline))
        save(root/'generation.json',proof['stats'])
        if fault == 'exception_after_bundle': raise RuntimeError('controlled exception after complete bundle')
        if fault == 'wrong_invocation': built['invocation'] = 'another-request'
        save(root/'built.json',built)
        if fault == 'rebind_checker_context':
            import json
            name = 'code/scoped_source/sparse_check.py'
            (root/'bundle'/name).write_text('def check(*a,**k): return {"status":"forged"}')
            manifest = load(root/'bundle/manifest.json'); manifest['files'][name] = sha(root/'bundle'/name)
            (root/'bundle/manifest.json').write_text(json.dumps(manifest))
            built['sha256'] = sha(root/'bundle/manifest.json'); (root/'built.json').write_text(json.dumps(built))
            inv['checker_sources'][name] = manifest['files'][name]
            (root/'invocation.json').write_text(json.dumps(inv))
    elif phase == 'check':
        built = load(root/'built.json',limit=1024**2)
        if built['invocation'] != inv['invocation']: raise ValueError('build invocation')
        manifest = load(root/'bundle/manifest.json',built['sha256'],limit=1024**2)
        # Pin EVERY mathematical checker, not producer-chosen self-consistent hashes.
        bind_checker(root,manifest,inv['checker_sources'])
        if fault == 'partial_output':
            with (root/'check.stdout').open('x') as stream: stream.write('{"partial":')
            raise RuntimeError('controlled partial checker output')
        # Replace this owned worker: no transient checker grandchild whose stale
        # pre-poll RSS snapshot could be mistaken for a surviving descendant.
        events.emit('ENTER',operation='independent_check_exec')
        with (root/'check.stdout').open('xb') as stream:
            os.dup2(stream.fileno(),1)
        os.execv(PYTHON,[PYTHON,'-B','-I','-S',str(root/'bundle/verify.py'),
            '--manifest-sha',built['sha256'],'--source-sha',spec['source_sha256'],
            '--deadline',str(deadline)])
    elif phase == 'receive':
        accepted = events.call('candidate_receive',lambda:receive(root,spec,inv['invocation'],invocation_sha))
        if fault == 'late_publish':
            save(root/'unaccepted_partial.json',accepted); time.sleep(10)
        events.call('candidate_publish',lambda:save(root/'accepted.json',accepted))
    else: raise ValueError('phase')
    tick(deadline)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase'); p.add_argument('root',type=Path)
    p.add_argument('--deadline',required=True,type=float); p.add_argument('--invocation-sha',required=True); a = p.parse_args()
    run(a.phase,a.root,a.deadline,a.invocation_sha)
