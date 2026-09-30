"""Owned H2 synthetic worker; a shared deadline, never a real-input selector."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scoped_proof.io import ROOT, PYTHON, Events, load, save, sha, tick
from source_enclosure.format import identity
from scoped_source.endpoint_supervised import validate_spec, receive, bind_checker, bind_producer


def run(phase, root, deadline, invocation_sha, checker_stdout_sha=None):
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
            subprocess.Popen([PYTHON,'-B','-c','import time; time.sleep(10)']); return
        if fault == 'memory':
            large = bytearray(128*2**20); time.sleep(10); return
        if spec['schema'] == 'H2_CAPTURE_CONTROL_V1':
            def create_object():
                from scoped_source.endpoint_intake import model_fixture
                return model_fixture()
            model,center,request = events.call('model_creation_imports',create_object)
            def take():
                from scoped_source.sparse_intake import capture_bound
                if fault == 'capture_delay':
                    save(root/'capture_partial.json',{'accepted':False,'source_sha256':spec['source_sha256']})
                    time.sleep(10)
                if fault == 'capture_exception': raise RuntimeError('controlled H2 capture failure')
                if fault in ('mutate_model','mutate_input'):
                    import torch
                    with torch.no_grad():
                        if fault == 'mutate_model': next(model.parameters()).add_(1)
                        else: center.add_(.25)
                return capture_bound(model,center,request,expected_source_sha256=spec['source_sha256'],deadline=deadline)
            doc = events.call('capture_and_validate',take)
            from scoped_source.endpoint_intake import receipt
            events.call('capture_receipt_publication',lambda:save(root/'model_intake.json',
                receipt(doc,inv['invocation'],inv['producer_sources'])))
            del model,center
        else:
            def create():
                from scoped_source.endpoint_source_controls import cases
                return next(doc for name,doc,reuse in cases() if name == spec['case'])
            doc = events.call('source_creation_imports', create)
        if identity(doc) != spec['source_sha256']: raise ValueError('frozen synthetic source')
        events.call('source_publication', lambda:save(root/'declared_source.json',doc))
        def importer():
            from act.back_end.solver.lp_certificate import propose
            return propose
        native = events.call('proposal_imports',importer)
        (root/'candidates').mkdir(); attempted = 0
        def propose(lp, **kwargs):
            nonlocal attempted
            tick(deadline); index = attempted; attempted += 1
            if fault == 'proposal_delay':
                save(root/'proposal_partial.json',{'lp_sha256':identity(lp),'accepted':False})
                time.sleep(10)
            if fault == 'proposal_exception': raise RuntimeError('controlled proposal failure')
            certificate = events.call('native_candidate_'+str(index),lambda:native(lp,**kwargs))
            events.call('candidate_publication_'+str(index),lambda:save(root/'candidates'/f'{index:05d}.json',
                {'lp_sha256':identity(lp),'certificate':certificate,'invocation':inv['invocation']}))
            tick(deadline); return certificate
        from scoped_source.endpoint_source_build import build
        reuse = [(tuple(p), j) for p,j in spec['reuse']]
        proof = events.call('source_construct_and_propose',lambda:build(doc,
            expected_source_sha256=spec['source_sha256'], deadline=deadline,
            mode=spec['mode'],reuse_keys=reuse,proposer=propose))
        if fault == 'missing_certificate':
            record = proof['proof']['duties'][0]
            if spec['mode'] == 'endpoints': record['endpoints'][0]['certificate'] = None
            else: record['certificate'] = None
        if fault == 'missing_both':
            for endpoint in proof['proof']['duties'][0]['endpoints']: endpoint['certificate'] = None
        if fault == 'missing_endpoint': proof['proof']['duties'][0]['endpoints'].pop()
        if fault == 'omit_property': proof['proof']['duties'].pop()
        if fault == 'wrong_mode': proof['mode'] = 'mccormick' if spec['mode']=='endpoints' else 'endpoints'
        if fault == 'serialization_delay':
            with (root/'serialize_partial.json').open('x') as stream: stream.write('{"unaccepted":')
            time.sleep(10)
        from scoped_source.endpoint_portable import pack
        built = events.call('serialize_and_bundle',lambda:pack(root/'bundle',doc,proof,
            spec['source_sha256'],inv['invocation'],deadline))
        save(root/'generation.json',{'stats':proof['proposal_stats'],'errors':proof['proposal_errors']})
        if fault == 'exception_after_bundle': raise RuntimeError('controlled exception after complete bundle')
        if fault == 'wrong_invocation': built['invocation'] = 'another-request'
        save(root/'built.json',built)
        if fault == 'rebind_checker_context':
            name = 'code/scoped_source/endpoint_source_check.py'
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
        bind_checker(root,manifest,inv['checker_sources'])
        if fault == 'partial_output':
            with (root/'check.stdout').open('x') as stream: stream.write('{"partial":')
            raise RuntimeError('controlled partial checker output')
        events.emit('ENTER',operation='independent_check_exec')
        with (root/'check.stdout').open('xb') as stream: os.dup2(stream.fileno(),1)
        os.execv(PYTHON,[PYTHON,'-B','-I','-S',str(root/'bundle/verify.py'),
            '--manifest-sha',built['sha256'],'--source-sha',spec['source_sha256'],
            '--mode',spec['mode'],'--deadline',str(deadline)])
    elif phase == 'receive':
        if fault == 'rewrite_check_stdout':
            checked = load(root/'check.stdout'); result = checked['result']
            result['positive'] = result['required']; result['status'] = 'CHECKED_DECLARED_SOURCE_POSITIVE'
            result['missing'] = 0
            for row in result['duties']: row.update(lower_bound='1',positive=True)
            # Internally consistent summaries, unchanged proof/source/invocation.
            (root/'check.stdout').write_text(json.dumps(checked))
        accepted = events.call('candidate_receive',lambda:receive(root,spec,inv['invocation'],invocation_sha,checker_stdout_sha))
        if fault == 'late_publish':
            save(root/'unaccepted_partial.json',accepted); time.sleep(10)
        events.call('candidate_publish',lambda:save(root/'accepted.json',accepted))
    else: raise ValueError('phase')
    tick(deadline)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase'); p.add_argument('root',type=Path)
    p.add_argument('--deadline',required=True,type=float); p.add_argument('--invocation-sha',required=True)
    p.add_argument('--checker-stdout-sha'); a=p.parse_args()
    run(a.phase,a.root,a.deadline,a.invocation_sha,a.checker_stdout_sha)
