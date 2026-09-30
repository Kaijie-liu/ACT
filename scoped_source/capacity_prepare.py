"""Hard-budget source-identity preparation, never a positive proof verdict.

One owned process creates/captures/validates/packs a fixed synthetic model.
The parent freezes only an observed completed result; failed attempts remain.
Timed verification must recreate everything and use these pre-bound identities.
"""
import argparse
import math
import os
from pathlib import Path
import time
import uuid
from scoped_proof.io import ROOT,PYTHON,Events,load,save,sha,tick
from scoped_proof.supervisor import execute
from scoped_source.capacity_intake import protocol,fixture,sources,expected_request,PROTOCOL_SHA
from source_enclosure.format import identity

FAULTS=('', 'delay','exception','partial','wrong_recipe','wrong_source','late','receive_delay')


def validate_spec(spec):
    protocol()
    if (set(spec)!={'case','control','recipe_sha256'} or spec['case'] not in ('tiny_control','full_size') or
            spec['control'] not in FAULTS or spec['recipe_sha256']!=PROTOCOL_SHA or
            spec['case']=='full_size' and spec['control']): raise ValueError('fixed preparation specification')


def worker(root,invocation_sha,deadline):
    inv=load(root/'invocation.json',invocation_sha); spec=load(root/'spec.json'); validate_spec(spec)
    if inv['spec_sha256']!=identity(spec) or inv['sources']!=sources() or deadline!=inv['produce_deadline']:
        raise ValueError('preparation identity/deadline')
    tick(deadline); events=Events(root,'prepare',inv['start'])
    if spec['control']=='delay':
        save(root/'delay_partial.json',{'accepted':False}); time.sleep(10)
    if spec['control']=='exception': raise RuntimeError('controlled preparation exception')
    model,center,request=events.call('create_model_and_imports',lambda:fixture(spec['case']))
    def capture():
        from scoped_source.sparse_intake import validate_object
        from scoped_source.capture import capture
        from scoped_source.graph import validate
        validate_object(model); tick(deadline)
        doc=capture(model,center,deadline=deadline,**request)
        validate(doc,identity(doc),lambda:tick(deadline)); return doc
    doc=events.call('capture_and_validate',capture); del model,center
    expected_request(spec['case'],doc['request']); declared_sha=identity(doc)
    def pack():
        from scoped_source.factored_source import pack
        return pack(doc,root/'source',lambda:tick(deadline),protocol()['source_chunk_bytes'])
    source_sha=events.call('chunk_and_publish',pack)
    from scoped_source.factored_io import inventory
    m=load(root/'source/manifest.json',limit=4*2**20)
    members={'manifest.json'}|{c['file'] for t in m['tensors'] for c in t['chunks']}
    source_bytes=inventory(root/'source',members,256*2**20)
    import torch
    result={'schema':'H2_CAPACITY_PREPARED_SOURCE_V1','invocation':inv['invocation'],
        'case':spec['case'],'recipe_sha256':PROTOCOL_SHA,'declared_source_sha256':declared_sha,
        'source_manifest_sha256':source_sha,'request':doc['request'],'request_sha256':identity(doc['request']),
        'source_bytes':source_bytes,'chunks':sum(len(t['chunks']) for t in m['tensors']),
        'torch_version':str(torch.__version__),'python_version':os.sys.version,
        'checkpoint_loaded':False,'dataset_loaded':False,'forward_executed':False,'solves':0,
        'real_capacity_admitted':False,'proof_status':'NOT_A_PROOF'}
    if spec['control']=='wrong_recipe': result['recipe_sha256']='0'*64
    if spec['control']=='wrong_source': result['declared_source_sha256']='0'*64
    if spec['control']=='partial':
        save(root/'unaccepted_partial.json',result); raise RuntimeError('controlled partial identity')
    events.call('candidate_publication',lambda:save(root/'prepared.json',result))
    if spec['control']=='late':
        save(root/'late_partial.json',{'accepted':False}); time.sleep(10)
    tick(deadline)


def receive(root,spec,inv,deadline,candidate_sha):
    """Identity preparation reception, not an independent network proof."""
    validate_spec(spec); tick(deadline); path=root/'prepared.json'; digest=candidate_sha
    from scoped_source.factored_io import required_hash
    required_hash(digest)
    value=load(path,digest,limit=2**20)
    if (value['schema']!='H2_CAPACITY_PREPARED_SOURCE_V1' or value['invocation']!=inv['invocation'] or
        value['case']!=spec['case'] or value['recipe_sha256']!=PROTOCOL_SHA or
        value['request_sha256']!=identity(value['request']) or value['proof_status']!='NOT_A_PROOF' or
        value['real_capacity_admitted'] is not False or value['solves']!=0 or
        any(value[k] is not False for k in ('checkpoint_loaded','dataset_loaded','forward_executed'))):
        raise ValueError('unbound or overclaimed prepared identity')
    expected_request(spec['case'],value['request'])
    required_hash(value['declared_source_sha256']); required_hash(value['source_manifest_sha256'])
    from scoped_source.capacity_sourcecheck import check_source
    checked=check_source(root/'source',value,spec['case'],lambda:tick(deadline))
    tick(deadline); return {'candidate_sha256':digest,'candidate':value,'source_check':checked}


def receive_worker(root,invocation_sha,deadline,candidate_sha):
    inv=load(root/'invocation.json',invocation_sha); spec=load(root/'spec.json'); validate_spec(spec)
    if inv['spec_sha256']!=identity(spec) or inv['sources']!=sources() or deadline!=inv['work_deadline']:
        raise ValueError('preparation reception identity/deadline')
    if spec['control']=='receive_delay': time.sleep(10)
    events=Events(root,'receive',inv['start'])
    result=events.call('independent_source_identity',lambda:receive(root,spec,inv,deadline,candidate_sha))
    events.call('reception_publication',lambda:save(root/'received.json',result)); tick(deadline)


def prepare(root,case='tiny_control',control='',*,budget=300.,rss_limit=2*2**30):
    start=time.monotonic(); cfg=protocol()
    if (type(budget) not in (int,float) or not math.isfinite(budget) or not 0<budget<=cfg['budget_seconds'] or
        type(rss_limit) is not int or not 0<rss_limit<=cfg['rss_limit_bytes']): raise ValueError('preparation limits')
    spec={'case':case,'control':control,'recipe_sha256':PROTOCOL_SHA}; validate_spec(spec)
    root=Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('output outside project')
    root.mkdir(parents=True,exist_ok=False); deadline=start+budget; work_deadline=deadline-min(1.,budget/5)
    produce_deadline=work_deadline-min(60.,budget/3)
    save(root/'spec.json',spec); inv={'schema':'H2_CAPACITY_PREP_INVOCATION_V1','invocation':uuid.uuid4().hex,
        'spec_sha256':identity(spec),'sources':sources(),'start':start,'deadline':deadline,
        'work_deadline':work_deadline,'produce_deadline':produce_deadline,'budget':budget,'rss_limit':rss_limit}
    binding=save(root/'invocation.json',inv)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    stages=[]; status='ERROR'; accepted=None; error=None; candidate_sha=None; receipt_sha=None
    try:
        for phase in ('prepare','receive'):
            load(root/'invocation.json',binding['sha256'])
            if inv['sources']!=sources(): raise ValueError('preparation implementation changed')
            phase_deadline=produce_deadline if phase=='prepare' else work_deadline
            command=[PYTHON,'-B']+(['-S'] if phase=='receive' else [])+[
                '-m','scoped_source.capacity_prepare','worker' if phase=='prepare' else 'receive',str(root),
                '--invocation-sha',binding['sha256'],'--deadline',str(phase_deadline)]
            if phase=='receive': command+=['--candidate-sha',candidate_sha]
            began=time.monotonic()-start
            stage=execute(command,root/(phase+'.log'),phase_deadline,env,rss_limit)
            stage.update(phase=phase,start_seconds=began,end_seconds=time.monotonic()-start)
            stages.append(stage); save(root/(phase+'_stage.json'),stage)
            status=stage['status']
            if status!='COMPLETED': break
            if phase=='prepare':
                if (root/'prepared.json').is_symlink() or (root/'prepared.json').stat().st_size>2**20:
                    raise ValueError('prepared candidate size/path')
                candidate_sha=sha(root/'prepared.json')
            else:
                receipt_sha=sha(root/'received.json'); accepted=load(root/'received.json',receipt_sha,limit=2**20)
                if accepted['candidate_sha256']!=candidate_sha or sha(root/'prepared.json')!=candidate_sha:
                    raise ValueError('parent-anchored candidate changed')
                if accepted['candidate']['invocation']!=inv['invocation']: raise ValueError('reception invocation')
                status='IDENTITY_PREPARED'
        load(root/'invocation.json',binding['sha256'])
        if inv['sources']!=sources(): raise ValueError('preparation implementation changed')
    except Exception as exc: status='ERROR'; error=repr(exc); accepted=None
    if time.monotonic()>=work_deadline: status='TIMEOUT'; accepted=None
    before=time.monotonic()-start
    terminal={'schema':'H2_CAPACITY_PREP_TERMINAL_V1','invocation_sha256':binding['sha256'],
        'invocation':inv['invocation'],'spec_sha256':identity(spec),'stages':stages,'accepted':accepted,
        'candidate_sha256':candidate_sha,'receipt_sha256':receipt_sha,
        'status':status,'error':error,'seconds_before_publication':before,
        'overhead_seconds':before-sum(s['seconds'] for s in stages),'proof_status':'NOT_A_PROOF'}
    term=save(root/'terminal.json',terminal)
    finish={'schema':'H2_CAPACITY_PREP_FINISH_V1','terminal_sha256':term['sha256'],
        'invocation':inv['invocation'],'status':status,'seconds_before_finish':time.monotonic()-start}
    marker=save(root/'finish.json',finish)
    elapsed=time.monotonic()-start
    if elapsed>=budget:
        status='TIMEOUT'; save(root/'publication_timeout.json',{'seconds':time.monotonic()-start}); elapsed=time.monotonic()-start
    return {'status':status,'seconds':elapsed,'root':str(root),'invocation':inv['invocation'],
            'finish_sha256':marker['sha256'],'real_capacity_admitted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=('worker','receive','prepare'))
    p.add_argument('root',type=Path); p.add_argument('--case',choices=('tiny_control','full_size'),default='full_size')
    p.add_argument('--invocation-sha'); p.add_argument('--deadline',type=float); p.add_argument('--candidate-sha'); a=p.parse_args()
    if a.action=='worker': worker(a.root,a.invocation_sha,a.deadline)
    elif a.action=='receive': receive_worker(a.root,a.invocation_sha,a.deadline,a.candidate_sha)
    else:
        result=prepare(a.root,a.case); save(a.root/'caller_observation.json',result); print(result)
