"""Bind the 20 selected S2a checkpoints and run the approved evaluation queue.

Preparation requires finished validation-only training. It never trains or
selects a model from test results. Execution reuses the S1 request supervisor.
"""
import time
ENTRY=time.monotonic()
import argparse
import os
from pathlib import Path
import resource
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,sha
from architecture_s1_execution import Artifacts,FILES,digest,finite_queue,require
import genbab_cpu_ready as backend

TRAIN=Path('/data1/Kane/MOE/results/architecture_s2a_training_20261031_r1')
TRAIN_FREEZE=Path('/data1/Kane/MOE/executions/architecture_s2a_training_20261031_r1/docs/architecture_s2a_training_freeze_20261031.json')
TRAIN_COMMIT='8b59db610e32dbcd3c3918cedfcb23b28c9fa949'
ARMS=('D-act','D-tot','M','M-reg')


def selected_epoch(epochs):
    require([r['epoch'] for r in epochs]==list(range(1,101)),'all 100 training epochs')
    require(all(r['total']==45000 and r['validation']['total']==5000 for r in epochs),
            'registered train/validation denominators')
    return max(epochs,key=lambda r:(r['validation']['certified'],r['validation']['clean'],-r['epoch']))


def models_from_training():
    import torch
    from act.pipeline.moe.architecture_s2_training import tune_items,selected_followups
    spec=load(TRAIN_FREEZE);completed=load(TRAIN/'complete.json')
    require(completed['status']=='TRAINING_COMPLETE' and completed['training_runs']==24
            and completed['selected_models']==20 and completed['test_evaluated'] is False,
            'all registered training completed without test selection')
    require(sha(TRAIN/'split.json')==spec['split_sha256'],'same 45k/5k split')
    require(load(TRAIN/'start.json')['immutable_commit']==TRAIN_COMMIT,'immutable training source')
    tuned=[load(TRAIN/item['id']/'complete.json') for item in tune_items(spec)]
    selected,followups=selected_followups(spec,tuned)
    expected=[selected[a] for a in ARMS]+[load(TRAIN/i['id']/'complete.json') for i in followups]
    require(len(completed['models'])==20 and
            {r['item']['id']:r for r in expected}=={r['item']['id']:r for r in completed['models']},
            'twenty validation-selected models, independent of worker completion order')
    for item in tune_items(spec)+followups:
        folder=TRAIN/item['id'];row=load(folder/'complete.json')
        require(row['item']==item and row['test_evaluated'] is False,'registered training attempt')
        best=selected_epoch([load(p) for p in sorted(folder.glob('epoch_*.json'))])
        require(row['best_epoch']==best['epoch'] and row['validation']==best['validation'],
                'validation-only checkpoint selection')
    models=[]
    for row in expected:
        item=row['item'];require(sha(row['checkpoint'])==row['checkpoint_sha256'],'selected weight bytes')
        payload=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
        require(payload['item']==item and payload['architecture']==item['architecture']
                and payload['seed']==item['seed'] and payload['code_commit']==TRAIN_COMMIT
                and payload['manifest_sha256']==sha(TRAIN_FREEZE)
                and payload['split_sha256']==spec['split_sha256']
                and payload['validation']==row['validation'] and payload['best_epoch']==row['best_epoch']
                and payload['test_evaluated'] is False,'training/checkpoint identity')
        require(sum(v.numel() for v in payload['state_dict'].values())==item['parameter_count'],
                'registered capacity')
        models.append(dict(item,id=f'{item["arm"]}_seed{item["seed"]}',
            training_id=item['id'],checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],
            best_epoch=row['best_epoch'],validation=row['validation']))
    require({(m['arm'],m['seed']) for m in models}=={(a,s) for a in ARMS for s in range(5)},
            'all four arms and five seeds')
    return sorted(models,key=lambda m:(m['seed'],ARMS.index(m['arm'])))


def all_requests(models):
    require(len(models)==20 and {(m['arm'],m['seed']) for m in models}==
            {(a,s) for a in ARMS for s in range(5)},'twenty paired models')
    require(len({m['id'] for m in models})==20,'unique model identities')
    ordered=sorted(models,key=lambda m:(m['seed'],ARMS.index(m['arm'])))
    rows=[dict(id='primary_'+m['id'],mode='primary',model=m['id'],radius=2) for m in ordered]
    for mode,count in [('incomplete',1000),('complete',200)]:
        for index in range(count):
            # Cyclic rotation is independent of any clean or certification result.
            turn=index%len(ordered)
            for m in ordered[turn:]+ordered[:turn]:
                rows.append(dict(id=f'{mode}_{m["id"]}_{index:04d}',mode=mode,
                                 model=m['id'],index=index,radius=2))
    return rows


def source_files():
    additional=('scripts/architecture_s2_study.py','scripts/architecture_s2_primary.py',
        'scripts/architecture_s2_audit.py','scripts/architecture_s1_audit.py',
        'act/pipeline/moe/architecture_statistics.py',
        'scripts/multilayer_train.py','scripts/architecture_s1_study.py',
        'act/pipeline/moe/architecture_s2_evaluation.py','act/pipeline/moe/architecture_s2_statistics.py',
        'act/pipeline/moe/architecture_s2_training.py','act/pipeline/moe/route_envelope_ibp.py',
        'act/back_end/moe/class_separated_top1.py','act/back_end/moe/genbab_pair.py',
        'act/back_end/moe/static_pair.py')
    return {p:sha(ROOT/p) for p in FILES+additional}


def prepare(root,workers):
    from multilayer_train import setup
    began=ENTRY;torch=setup(1);require(1<=workers<=8,'worker ceiling')
    # Must fail before test materialization if training is incomplete.
    models=models_from_training();training=load(TRAIN_FREEZE)
    from torchvision.datasets import CIFAR10
    import numpy as np
    from act.pipeline.moe.architecture_s2_evaluation import pack_arrays,check_inputs
    data=CIFAR10(training['dataset_root'],train=False,download=False)
    arrays=dict(pixels=np.ascontiguousarray(data.data.transpose(0,3,1,2)),
                labels=np.asarray(data.targets,dtype=np.int64))
    check_inputs(arrays)
    store=Artifacts.create(root)
    inputs=store.root/'inputs.json.gz';input_sha=store.write(inputs,pack_arrays(arrays))
    external=backend.source_identity();requests=all_requests(models)
    spec=dict(schema='S2A_EVALUATION_V1',study='S2A',models=models,source_files=source_files(),
        training_manifest=dict(path=str(TRAIN_FREEZE),sha256=sha(TRAIN_FREEZE)),
        training_result=dict(path=str(TRAIN/'complete.json'),sha256=sha(TRAIN/'complete.json')),
        inputs=dict(path=str(inputs),sha256=input_sha,indices=list(range(10000)),
            raw_test_sha256=sha(Path(training['dataset_root'])/'cifar-10-batches-py/test_batch'),
            convention='uint8 -> float32 /255, clipped float32 2/255 box; secondary exact float64 lift'),
        backend_sources=external,backend_sources_sha256=digest(external),
        primary_torch_version=str(torch.__version__),primary_batch_size=256,
        complete_configuration=backend.CONFIG,workers=workers,worker_threads=1,
        request_seconds=300,timing_reserve=[12,10,5],artifact_bytes=128*2**20,as_bytes=8*2**30,
        primary_models=20,incomplete_calls=20000,complete_calls=4000,
        request_catalog_sha256=digest(requests),
        queue_order='primary, incomplete, complete; all inputs including clean errors; cyclic models',
        time_amendment='user20261031: finish fixed queue; 40h not truncation; summed and wall reported',
        bootstrap=dict(resamples=10000,seed=20261031,unit='crossed paired seeds and inputs'),
        gates=dict(m_minus_active_pp=2,active_ci_lower_strictly_positive=True,
                   m_minus_total_nonnegative=True,m_single_route_fraction=.9),
        preparation_seconds=time.monotonic()-began,source_complete_proof=False)
    path=store.root/'manifest.json';store.write(path,spec)
    print('FROZEN',path,'primary20 / incomplete20000 / complete4000')


def run(root):
    from architecture_s1_study import resources
    require(ROOT!=Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(),'immutable execution copy')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    store=Artifacts(root);manifest=store.root/'manifest.json';spec=load(manifest)
    require(all(sha(ROOT/p)==h for p,h in spec['source_files'].items()),'frozen S2 evaluation code')
    require(digest(backend.source_identity())==spec['backend_sources_sha256'],'frozen common external engine')
    require(sha(spec['inputs']['path'])==spec['inputs']['sha256'],'shared physical input source')
    for m in spec['models']:require(sha(m['checkpoint'])==m['checkpoint_sha256'],'all selected weights')
    observation=resources(spec['workers'])
    require(observation['ordinary_available_disk_bytes']>=2*2**30,'existing disk admission')
    require(observation['load'][0]+spec['workers']<=observation['allowed_cpus'],'existing CPU admission')
    require(observation['memory']['MemAvailable']>=(8*spec['workers']+4)*2**30,'existing memory admission')
    store.write(store.root/'start.json.gz',dict(commit=(ROOT/'.frozen_commit').read_text().strip(),
        manifest_sha256=sha(manifest),resources=observation,pid=os.getpid(),clock=time.time(),
        startup_and_admission_seconds=time.monotonic()-ENTRY))
    result=finite_queue(store.root,all_requests(spec['models']),manifest,workers=spec['workers'],
        seconds=300,timing_reserve=tuple(spec['timing_reserve']))
    print(result)
    # Record-only analysis follows the settled queue. Its elapsed time is
    # separate and cannot turn a late verifier output into an online positive.
    from architecture_s2_audit import analyze
    report=analyze(store.root)
    store.write(store.root/'evaluation_summary.json.gz',report,terminal=True)
    print(report['status'],report['registered_decision'],report['online_states'])
    return 0 if result['status']=='COMPLETE' else 1


def primary_report(root):
    """Summarize accepted records only; never add an offline success."""
    import numpy as np
    from architecture_s1_execution import read_record
    from act.pipeline.moe.architecture_s2_evaluation import unpack_arrays,check_inputs,check_primary_arrays
    from act.pipeline.moe.architecture_s2_statistics import primary_summary
    store=Artifacts(root);manifest=store.root/'manifest.json';spec=load(manifest)
    require(all(sha(ROOT/p)==h for p,h in spec['source_files'].items()),'frozen report code')
    require(digest(all_requests(spec['models']))==spec['request_catalog_sha256'],'complete paired roster')
    physical=unpack_arrays(read_record(spec['inputs']['path']));check_inputs(physical)
    require(sha(spec['inputs']['path'])==spec['inputs']['sha256'],'physical input hash')
    certified=np.zeros((4,5,10000),dtype=bool);clean=np.zeros_like(certified)
    counts=np.zeros(certified.shape,dtype=np.uint8);sources=[]
    for m in spec['models']:
        request_id='primary_'+m['id'];folder=store.root/request_id
        ledger=read_record(store.root/(request_id+'.ledger.json.gz'))
        require(ledger['accepted'] and ledger['status']=='AUXILIARY_COMPLETE',
                'all 20 primary measurements must already be accepted online')
        require(sha(folder/'terminal.json.gz')==ledger['terminal_sha256'],'online terminal hash')
        terminal=read_record(folder/'terminal.json.gz')
        require(terminal['accepted'] and terminal['status']=='AUXILIARY_COMPLETE','no offline upgrade')
        query=read_record(folder/'request.json.gz');publication=read_record(folder/'published.json.gz')
        outcome=read_record(folder/'outcome.json.gz');value=outcome['value']
        require(query['manifest_sha256']==sha(manifest) and query['model']==m['id']
                and query['mode']=='primary' and query['radius']==2,'fixed primary identity')
        require(outcome['request_sha256']==sha(folder/'request.json.gz')
                and publication['sha256']==sha(folder/'outcome.json.gz')
                and outcome['clock']<=publication['clock']<query['worker_deadline'],
                'original publication identity and cutoff')
        require(value['checkpoint_sha256']==m['checkpoint_sha256']
                and value['input_sha256']==spec['inputs']['sha256']
                and value['family']==m['family'] and value['count']==10000 and value['properties']==9,
                'same trained source and full primary denominator')
        require(sha(folder/'arrays.json.gz')==value['array_sha256'],'raw primary arrays')
        arrays=unpack_arrays(read_record(folder/'arrays.json.gz'))
        check_primary_arrays(arrays,family=m['family'],count=10000)
        require(np.array_equal(arrays['label'],physical['labels']),'all test labels in canonical order')
        require(value['clean']==int(arrays['clean'].sum()) and
                value['certified']==int(arrays['certified'].sum()),'same online primary counts')
        a,s=ARMS.index(m['arm']),m['seed']
        certified[a,s]=arrays['certified'];clean[a,s]=arrays['clean'];counts[a,s]=arrays['route_count']
        sources.append(dict(id=request_id,array_sha256=value['array_sha256'],
                            ledger_sha256=sha(store.root/(request_id+'.ledger.json.gz')),
                            request_seconds=ledger['seconds'],batches=value['batches']))
    result=primary_summary(certified,clean,counts,**{
        k:spec['bootstrap'][k] for k in ('resamples','seed')})
    queue=read_record(store.root/'queue_complete.json.gz') if (store.root/'queue_complete.json.gz').exists() else None
    result.update(source_manifest_sha256=sha(manifest),sources=sources,
        audit_scope='record identities/coverage/statistics only; no recomputed network or source-complete proof',
        secondary_stage_complete=bool(queue and queue['status']=='COMPLETE'
            and queue['settled']==24020 and queue['not_executed']==0))
    store.write(store.root/'primary_summary.json.gz',result)
    print(result['decision'],result['comparisons'],result['gates'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','primary-report'])
    parser.add_argument('root',type=Path);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root,args.workers)
    elif args.command=='primary-report':primary_report(args.root)
    else:raise SystemExit(run(args.root))
