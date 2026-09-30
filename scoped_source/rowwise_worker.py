"""Owned row-wise H2 worker; fixed controls and one shared clock only."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from scoped_proof.io import ROOT, PYTHON, Events, load, save, sha, tick
from source_enclosure.format import identity
from scoped_source.rowwise_supervised import validate_spec,receive,bind_checker,bind_producer,required_sha,protocol


def run(phase,root,deadline,invocation_sha,checker_stdout_sha=None):
    inv=load(root/'invocation.json',required_sha(invocation_sha)); spec=load(root/'spec.json'); validate_spec(spec)
    if identity(spec)!=inv['spec_sha256'] or deadline>inv['work_deadline']: raise ValueError('invocation/deadline identity')
    bind_producer(spec,inv); tick(deadline)
    events=Events(root,phase,inv['start']); fault=spec['control']
    def delay(name):
        save(root/(name+'_partial.json'),{'status':'UNACCEPTED_CONTROL_DELAY'}); time.sleep(10)
    if fault==phase+'_delay': delay(phase)
    if phase=='produce':
        if fault=='descendant': subprocess.Popen([PYTHON,'-B','-c','import time; time.sleep(10)']); return
        if fault=='memory':
            large=bytearray(128*2**20); time.sleep(10); return
        if spec['intake']=='model':
            def create():
                from scoped_source.endpoint_intake import model_fixture
                return model_fixture()
            model,center,request=events.call('model_creation_imports',create)
            def capture():
                from scoped_source.sparse_intake import capture_bound
                if fault=='capture_delay': delay('capture')
                if fault=='capture_exception': raise RuntimeError('controlled capture failure')
                if fault in ('mutate_model','mutate_input'):
                    import torch
                    with torch.no_grad():
                        if fault=='mutate_model': next(model.parameters()).add_(1)
                        else: center.add_(.25)
                return capture_bound(model,center,request,expected_source_sha256=spec['declared_source_sha256'],deadline=deadline)
            doc=events.call('capture_and_validate',capture)
            from scoped_source.endpoint_intake import receipt
            events.call('capture_receipt_publication',lambda:save(root/'model_intake.json',
                receipt(doc,inv['invocation'],inv['producer_sources'])))
            del model,center
        else:
            def create():
                from scoped_source.endpoint_source_controls import cases
                return next(doc for name,doc,reuse in cases() if name==spec['case'])
            doc=events.call('source_creation_imports',create)
        if identity(doc)!=spec['declared_source_sha256']: raise ValueError('frozen declared source')
        tree=root/'bundle/proof'; tree.mkdir(parents=True)
        def chunk():
            from scoped_source.factored_source import pack
            if fault=='chunk_delay': delay('chunk')
            return pack(doc,tree/'source',lambda:tick(deadline),protocol()['source_chunk_bytes'])
        source_sha=events.call('source_chunk_construction_publication',chunk)
        if source_sha!=spec['source_manifest_sha256']: raise ValueError('frozen chunked source')
        del doc
        def importer():
            from scoped_source.rowwise_native import propose
            return propose
        native=events.call('proposal_imports',importer); (root/'candidates').mkdir(); attempts=0
        def proposer(lp,**kwargs):
            nonlocal attempts
            if (set(kwargs)!={'time_limit'} or type(kwargs['time_limit']) not in (int,float) or
                    not math.isfinite(kwargs['time_limit']) or not 0<=kwargs['time_limit']<=300):
                raise ValueError('fixed proposer remaining-budget contract')
            # The builder's relative hint cannot create or extend a deadline.
            # All native validation, imports, solving and checking share this phase deadline.
            tick(deadline); i=attempts; attempts+=1
            if fault=='proposal_delay': delay('proposal')
            if fault=='proposal_exception': raise RuntimeError('controlled proposal failure')
            cert=events.call('native_candidate_'+str(i),lambda:native(lp,deadline=deadline))
            events.call('candidate_publication_'+str(i),lambda:save(root/'candidates'/f'{i:05d}.json',
                {'lp_sha256':identity(lp),'certificate':cert,'invocation':inv['invocation']}))
            tick(deadline); return cert
        def construct():
            from scoped_source.factored_build import build
            if fault=='construct_delay': delay('construct')
            return build(tree,expected_source_manifest=source_sha,mode=spec['mode'],
                reuse_keys=[(tuple(p),j) for p,j in spec['reuse']],deadline=deadline,proposer=proposer)
        proof_sha=events.call('source_construct_and_propose',construct)
        # Fault injections alter only this newly generated controlled attempt.
        # Rebind inner hashes so mathematical/coverage checks, not stale hashes,
        # must reject malformed obligations. No frozen artifact is overwritten.
        if fault in ('missing_certificate','missing_both','missing_endpoint','omit_property','omit_pair','wrong_mode'):
            m=load(tree/'manifest.json'); ref=m['pairs'][0]; part=load(tree/ref['file'])
            row=part['duties'][0]
            if fault=='missing_certificate':
                if spec['mode']=='endpoints': row['endpoints'][0]['certificate']=None
                else: row['certificate']=None
            if fault=='missing_both':
                for end in row['endpoints']: end['certificate']=None
            if fault=='missing_endpoint': row['endpoints'].pop()
            if fault=='omit_property': part['duties'].pop()
            if fault=='omit_pair': m['pairs'].pop()
            if fault=='wrong_mode': m['mode']='mccormick' if spec['mode']=='endpoints' else 'endpoints'
            raw=json.dumps(part,sort_keys=True).encode(); (tree/ref['file']).write_bytes(raw)
            ref.update(bytes=len(raw),sha256=sha(tree/ref['file']))
            (tree/'manifest.json').write_text(json.dumps(m,sort_keys=True)); proof_sha=identity(m)
        if fault=='serialization_delay': delay('serialization')
        from scoped_source.rowwise_portable import publish
        built=events.call('serialize_and_bundle',lambda:publish(root/'bundle',source_sha,proof_sha,
            spec['mode'],inv['invocation'],deadline))
        if fault=='exception_after_bundle': raise RuntimeError('controlled exception after bundle')
        if fault=='wrong_invocation': built['invocation']='another-request'
        save(root/'built.json',built)
        if fault=='rebind_checker_context':
            name='code/scoped_source/rowwise_check.py'
            (root/'bundle'/name).write_text('def check(*a,**k): return {"status":"forged"}')
            m=load(root/'bundle/manifest.json'); m['files'][name]={'sha256':sha(root/'bundle'/name),'bytes':(root/'bundle'/name).stat().st_size}
            (root/'bundle/manifest.json').write_text(json.dumps(m)); built['sha256']=sha(root/'bundle/manifest.json')
            (root/'built.json').write_text(json.dumps(built)); inv['checker_sources'][name]=m['files'][name]['sha256']
            (root/'invocation.json').write_text(json.dumps(inv))
    elif phase=='check':
        built=load(root/'built.json',limit=2**20)
        if built['invocation']!=inv['invocation']: raise ValueError('checker invocation')
        m=load(root/'bundle/manifest.json',required_sha(built['sha256']),limit=4*2**20)
        bind_checker(root,m,inv['checker_sources'])
        if fault=='partial_output':
            with (root/'check.stdout').open('x') as stream: stream.write('{"partial":')
            raise RuntimeError('controlled partial checker output')
        events.emit('ENTER',operation='independent_check_exec')
        with (root/'check.stdout').open('xb') as stream: os.dup2(stream.fileno(),1)
        os.execv(PYTHON,[PYTHON,'-B','-I','-S',str(root/'bundle/verify.py'),
            '--manifest-sha',built['sha256'],'--source-sha',spec['source_manifest_sha256'],
            '--proof-sha',built['proof_manifest_sha256'],'--mode',spec['mode'],'--deadline',str(deadline)])
    elif phase=='receive':
        if fault=='rewrite_check_stdout':
            value=load(root/'check.stdout'); result=value['result']
            result.update(positive=result['required'],status='CHECKED_DECLARED_SOURCE_POSITIVE',missing=0)
            for row in result['duties']: row.update(lower_bound='1',positive=True)
            (root/'check.stdout').write_text(json.dumps(value))
        accepted=events.call('candidate_receive',lambda:receive(root,spec,inv['invocation'],
            invocation_sha,checker_stdout_sha,deadline))
        if fault=='late_publish':
            save(root/'unaccepted_partial.json',accepted); time.sleep(10)
        events.call('candidate_publish',lambda:save(root/'accepted.json',accepted))
    else: raise ValueError('phase')
    tick(deadline)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase'); p.add_argument('root',type=Path)
    p.add_argument('--deadline',required=True,type=float); p.add_argument('--invocation-sha',required=True)
    p.add_argument('--checker-stdout-sha'); a=p.parse_args()
    run(a.phase,a.root,a.deadline,a.invocation_sha,a.checker_stdout_sha)
