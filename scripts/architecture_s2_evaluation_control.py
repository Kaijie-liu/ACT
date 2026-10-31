"""Finite synthetic check of the S2 common-engine adapter; no trained model.

Uses the existing owned-session supervisor, not a new execution framework.
"""
import time
ENTRY=time.monotonic()
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
from scoped_proof.io import load,save,sha
import genbab_cpu_ready as backend

FILES=('act/pipeline/moe/architecture_evaluation.py',
       'act/back_end/moe/multilayer_external.py','scripts/b2_observed_worker.py',
       'scripts/genbab_cpu_ready.py','scripts/architecture_s2_evaluation_control.py')


def worker(folder,deadline):
    fd=os.open(os.devnull,os.O_WRONLY);os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
    cases=[]
    try:
        backend.limits(deadline);environment=backend.probe()
        from act.util.typing_compat import install_typing_override
        install_typing_override()
        import torch
        from torch import nn
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
        from act.pipeline.moe.architecture_evaluation import evaluate,crown_ibp
        from act.back_end.moe.multilayer import RoutedLayer
        from act.back_end.moe.multilayer_external import SequentialTop2
        from multilayer_stage0 import affine
        from architecture_s1_execution import audit_value
        x=torch.zeros(1,1)
        tensors=dict(center=x,lower=x-1,upper=x+1,rows=torch.tensor([[1.,-1.]]),thresholds=torch.zeros(1))
        # f=.1+ReLU(x+.5)-2ReLU(x)+ReLU(x-.5)>=.1 on [-1,1].
        # The unsplit LP relaxation is loose; this checks real ReLU branching
        # through the SAME nonlinear/BBPS configuration used on MoE graphs.
        dense=nn.Sequential(affine([[1.],[1.],[1.]],[.5,0.,-.5]),nn.ReLU(),
                            affine([[1.,-2.,1.],[0.,0.,0.]],[.1,0.]))
        import copy
        def run(name,model,family,config):
            result=evaluate(model,tensors,family=family,mode='complete',deadline=deadline,
                            complete_configuration=config)
            cov=audit_value(result,family=family,experts=[3] if family=='moe' else [],properties=1)
            cases.append(dict(name=name,status=result['status'],coverage=cov,result=result))
            if result['status']!='POSITIVE':raise AssertionError((name,result['status']))
            if config is not None:
                if result['evaluation']['explicit_complete_configuration']!=config:raise AssertionError('configuration binding')
                if any(r['outcome']['branching']!='nonlinear' for r in result['histories']):raise AssertionError('common nonlinear configuration')
            return result
        config=dict(backend.CONFIG)
        dense_result=run('dense_common_real_branching',dense,'dense',config)
        if not any(r['outcome']['stage']=='nonlinear_bab' for r in dense_result['histories']):
            raise AssertionError('dense control must exercise actual branch-and-bound')
        moe=SequentialTop2([RoutedLayer(affine([[0.],[0.],[0.]],[0.,0.,0.]),
            [affine([[0.],[0.]],[2.,0.]) for _ in range(3)],k=2,mode='selected_softmax')])
        result=run('moe_common_all_tie_pairs',moe,'moe',config)
        if result['verified_histories']!=3:raise AssertionError('all tie-legal pairs')
        old=run('legacy_dense_default_unchanged',affine([[0.],[0.]],[2.,0.]),'dense',None)
        if old['histories'][0]['outcome']['branching']!='kfsb':raise AssertionError('old S1 default changed')
        unsafe=evaluate(affine([[0.],[0.]],[-1.,0.]),tensors,family='dense',mode='complete',
                        deadline=deadline,complete_configuration=config)
        if unsafe['status']!='UNSAFE_REPLAYED':raise AssertionError('full-model center replay')
        cases.append(dict(name='common_clean_error_replay',status=unsafe['status']))
        save(folder/'outcome.json',dict(status='PASS',cases=cases,config=config,
            source_files={p:sha(ROOT/p) for p in FILES},backend_source=environment['sources'],
            trained_model_calls=0,deadline=deadline,clock=time.monotonic(),seconds=time.monotonic()-ENTRY))
        return 0
    except BaseException as exc:
        save(folder/'error.json',dict(kind=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                                    completed_cases=cases,seconds=time.monotonic()-ENTRY))
        return 1


def run(folder):
    from scoped_proof.owned_session import execute
    folder.mkdir()
    save(folder/'freeze.json',dict(schema='S2_COMMON_ENGINE_SYNTHETIC_CONTROL_V1',seconds=300,
        source_files={p:sha(ROOT/p) for p in FILES},configuration=backend.CONFIG,
        cases=['dense analytic triangular positive, actual nonlinear BBPS BaB',
               'MoE all three tie-legal pairs positive, same configuration',
               'old S1 dense default kfsb unchanged','common-config clean error full replay'],
        trained_models=0,no_training_changes=True))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    session=execute([str(backend.ENVIRONMENT/'bin/python'),'-I','-B',str(Path(__file__).resolve()),
                     'worker',str(folder),str(ENTRY+288)],run_deadline=ENTRY+290,
                     cleanup_deadline=ENTRY+295,env=env,rss_limit=None,development_policy=backend.POLICY)
    save(folder/'session.json',session)
    value=load(folder/'outcome.json') if (folder/'outcome.json').exists() else None
    ok=(session['status']=='COMPLETED' and session['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
        and value is not None and value['status']=='PASS' and value['clock']<value['deadline']
        and time.monotonic()<ENTRY+300)
    record=dict(status='PASS' if ok else 'FAIL',seconds=time.monotonic()-ENTRY,
        cases=0 if value is None else len(value['cases']),session_status=session['status'],
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},trained_model_calls=0,
        timing_interpretation='control with training concurrently active; not performance evidence')
    save(folder/'complete.json',record);print(json.dumps(record))
    return 0 if ok else 1


def reception(folder):
    """Check the existing supervisor's S2 dispatch and online record receiver."""
    import copy
    import resource
    import signal
    from architecture_s1_execution import Artifacts,FILES as BASE_FILES,measured_invoke,read_record,digest
    from architecture_s2_primary import receive
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('300s integration control')))
    signal.setitimer(signal.ITIMER_REAL,max(.01,299-(time.monotonic()-ENTRY)))
    folder.mkdir()
    files=BASE_FILES+('scripts/architecture_s2_primary.py','scripts/multilayer_train.py',
        'act/pipeline/moe/route_envelope_ibp.py','act/pipeline/moe/architecture_s2_evaluation.py',
        'scripts/architecture_s2_evaluation_control.py')
    spec=dict(study='S2A',synthetic_control=True,primary_batch_size=4,
        source_files={p:sha(ROOT/p) for p in files},
        backend_sources_sha256=digest(backend.source_identity()),complete_configuration=backend.CONFIG,
        cases=['dense primary','E8 tie primary','dense common complete',
               'missing property rejection','changed bound source rejection'],seconds=300)
    save(folder/'freeze.json',spec);store=Artifacts.create(folder/'requests');rows=[]
    for request in [dict(id='dense_primary',mode='primary',family='dense'),
                    dict(id='moe_primary',mode='primary',family='moe'),
                    dict(id='dense_complete',mode='complete',case='positive')]:
        row=measured_invoke(store.root,request,folder/'freeze.json',seconds=60,timing_reserve=(8,5,2))
        rows.append(row);store.write(store.root/(request['id']+'.ledger.json.gz'),row,terminal=True)
        expected='AUXILIARY_COMPLETE' if request['mode']=='primary' else 'POSITIVE'
        if row['status']!=expected:
            save(folder/'complete.json',dict(status='FAIL',rows=rows,seconds=time.monotonic()-ENTRY))
            return 1
    target=store.root/'moe_primary';query=read_record(target/'request.json.gz')
    value=read_record(target/'outcome.json.gz')['value'];rejects=[]
    for key,wrong in [('properties',8),('bound_source_sha256','0'*64)]:
        changed=copy.deepcopy(value);changed[key]=wrong
        try:receive(target,query,spec,changed,ENTRY+295)
        except ValueError:rejects.append(key)
        else:raise AssertionError('invalid primary accepted: '+key)
    account=load(store.account)
    physical=sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file())
    if not physical<=account['used']<=account['cap']:raise AssertionError('shared byte account')
    record=dict(status='PASS',native_calls=len(rows),rejections=rejects,
        seconds=time.monotonic()-ENTRY,rows=rows,physical_bytes=physical,account=account,
        trained_model_calls=0,freeze_sha256=sha(folder/'freeze.json'),
        timing_interpretation='synthetic integration concurrent with training; not speed evidence')
    if record['seconds']>=300:raise TimeoutError('integration full cost')
    save(folder/'complete.json',record);signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps({k:record[k] for k in ('status','native_calls','rejections','seconds')}))
    return 0


if __name__=='__main__':
    if sys.argv[1]=='worker':raise SystemExit(worker(Path(sys.argv[2]),float(sys.argv[3])))
    if sys.argv[1]=='run':raise SystemExit(run(Path(sys.argv[2])))
    if sys.argv[1]=='reception':raise SystemExit(reception(Path(sys.argv[2])))
    raise SystemExit('run NEW_DIRECTORY')
