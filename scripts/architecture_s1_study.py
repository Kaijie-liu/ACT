"""S1 final model binding and finite evaluation launch; no new search policy."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha
from architecture_s1_execution import Artifacts, FILES, digest, catalog, pgd_catalog, finite_queue, require
import genbab_cpu_ready as backend

TRAIN = Path('/data1/Kane/MOE/results/architecture_s1_training_20261029_r1')
INPUTS = Path('/data1/Kane/MOE/results/architecture_s1_inputs_20261029_r1')


def selected_epoch(epochs):
    require([e['epoch'] for e in epochs] == list(range(1, 51)), 'all50 frozen epochs')
    require(all(e['validation']['total'] == 5000 for e in epochs), 'common validation denominator')
    return max(epochs, key=lambda e: (e['validation']['accuracy'], -e['epoch']))


def models_from_training():
    """No forward pass: training selection was completed independently."""
    import torch
    frozen = load(ROOT/'docs/architecture_s1_training_freeze_20261029.json')
    completed = load(TRAIN/'complete.json')
    require(completed['status'] == 'COMPLETE' and len(completed['models']) == 6, 'all6 dense seeds ready')
    require(sha(TRAIN/'split.json') == frozen['split_sha256'], 'common training split')
    old = load(ROOT/'docs/multilayer_training_freeze_20261022.json')
    result = []
    for item in frozen['models']:
        folder = TRAIN/item['id']; row = load(folder/'complete.json')
        require(row in completed['models'], 'training batch/model agreement')
        epochs = [load(p) for p in sorted(folder.glob('epoch_*.json'))]
        best = selected_epoch(epochs)
        require(best['epoch'] == row['best_epoch'] and best['validation'] == row['validation'],
            'earliest validation selection')
        require(sha(row['checkpoint']) == row['checkpoint_sha256'], 'selected dense weights')
        payload = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
        require(payload['architecture'] == item['architecture'] and payload['seed'] == item['seed']
            and payload['best_epoch'] == best['epoch'] and payload['validation'] == best['validation']
            and payload['test'] == row['test'] and payload['split_sha256'] == frozen['split_sha256']
            and payload['manifest_sha256'] == sha(ROOT/'docs/architecture_s1_training_freeze_20261029.json'),
            'dense source/recipe/checkpoint binding')
        require(payload['code_commit'] == 'f619c24bd7480edb3e50e47413c9048230a4f721', 'immutable training source')
        require(sum(v.numel() for v in payload['state_dict'].values()) == item['parameter_count']
            and payload['parameter_count'] == item['parameter_count'], 'dense real parameters')
        result.append(dict(**item, family='dense', checkpoint=row['checkpoint'],
            checkpoint_sha256=row['checkpoint_sha256'], selected_epoch=row['best_epoch'],
            test=row['test'], validation=row['validation'], training_completion_sha256=sha(folder/'complete.json'),
            epoch_records={p.name:sha(p) for p in sorted(folder.glob('epoch_*.json'))}))
        del payload
    for item in frozen['old_moe_models']:
        recipe = next(m for m in old['models'] if m['id'] == item['id'])
        require(sha(item['checkpoint']) == item['checkpoint_sha256'], 'old fixed MoE weights')
        payload = torch.load(item['checkpoint'], map_location='cpu', weights_only=True)
        require(payload['architecture'] == recipe['architecture'] and payload['seed'] == recipe['seed'],
            'old MoE source configuration')
        result.append(dict(**item, family='moe', seed=recipe['seed'], architecture=recipe['architecture'],
            group='L1' if recipe['architecture']['layers'] == 1 else 'L2',
            parameter_count=sum(v.numel() for v in payload['state_dict'].values())))
        del payload
    # Keep capacity comparators adjacent; cyclic per-input order is independent
    # of accuracy, route statistics and verification outcomes.
    order = [f'{name}_seed{seed}' for seed in range(3)
        for name in ('L1E8','Dense_total','Dense_active','L2E8')]
    require(set(order) == {m['id'] for m in result}, 'twelve frozen model identities')
    return [next(m for m in result if m['id'] == name) for name in order]


def all_requests(models):
    return pgd_catalog(models)+catalog(models, 'complete')+catalog(models, 'incomplete')


def source_files():
    return {p:sha(ROOT/p) for p in FILES+('scripts/architecture_s1_study.py',
        'act/back_end/moe/class_separated_top1.py', 'act/back_end/moe/genbab_pair.py',
        'act/back_end/moe/static_pair.py')}


def freeze(path):
    began = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    models = models_from_training()
    record = load(ROOT/'docs/architecture_s1_records_20261029.json')
    controls = {}
    for key in ('evaluation_adapter_control', 'request_execution_control', 'early_timeout_delta_control',
                'public_pgd_control'):
        row = record[key]
        controls[key] = row
        require(row['status'] == 'PASS', 'required S1 integration control '+key)
        path_key = row.get('root', row.get('path'))
        if path_key:
            outcome_path = Path(path_key)
            if outcome_path.is_dir():
                outcome_path /= 'complete.json'
            require(sha(outcome_path) == row['complete_sha256'], 'recorded control identity')
    identity = load(INPUTS/'catalog.json')
    require(sha(INPUTS/'inputs.pt') == identity['input_sha256'] and identity['indices'] == list(range(1000)),
        'fixed physical1000 input prefix')
    requests = all_requests(models)
    require(len(requests) == 31200 and len({r['id'] for r in requests}) == 31200, 'finite S1 denominator')
    source = backend.source_identity()
    input_bytes = sum(p.stat().st_size for p in INPUTS.iterdir() if p.is_file())
    input_receipts = [INPUTS.with_suffix('.freeze.json'), INPUTS.with_suffix('.session.json')]
    input_bytes += sum(p.stat().st_size for p in input_receipts)
    save(path, dict(schema='S1_NUMERICAL_EVALUATION_20261030_V1', models=models,
        source_files=source_files(), backend_sources=source, backend_sources_sha256=digest(source),
        genbab_config=backend.CONFIG, dense_override={
            'bab.branching.method':'kfsb','bab.branching.nonlinear_split.disable':True},
        inputs=dict(path=str(INPUTS/'inputs.pt'), sha256=sha(INPUTS/'inputs.pt'),
            catalog=str(INPUTS/'catalog.json'), catalog_sha256=sha(INPUTS/'catalog.json'),
            initial_account_bytes=input_bytes, indices='first1000; complete first200', historical_exposure=True),
        request_catalog_sha256=digest(requests), queue_order='publicPGD, complete, incomplete; model order cyclic by input/radius',
        counts=dict(native_batches=31200, complete_calls=4800, incomplete_items=24000,
            public_pgd_batches=2400, public_pgd_domains=24000),
        public_pgd=dict(steps=20,restarts=2,loss='minimum of all nine true-label margins',
            step_size='epsilon/4',seed='20261029+100000*model_seed+10000*(radius==8)+index',
            domain_seconds=20,batch_domains=10,loading='shared within auxiliary batch, separately metered',
            early_stop='first valid original-model counterexample; clean errors are counted',
            results_not_supplied_to_verifiers=True),
        request_seconds=300, timing_reserve=[12,10,5], workers=8, native_threads=1,
        per_process_address_space_bytes=8*2**30, artifact_bytes=128*2**20, resource_policy=backend.POLICY,
        memory_semantics='AS per process, not aggregate physical memory; RSS telemetry only',
        cpu=True, gpu=False, stop_new_admission_on_error=True, no_automatic_retries=True,
        runtime_cost='cold request includes startup/load/input/pruning/bounds/allhistories/replay/serialization/reception/cleanup/terminal',
        capacity_matches=['L1E8 vs Dense_total','L1E8 vs Dense_active'], secondary='L2E8',
        primary=dict(radius=2,denominator_per_model=200,endpoint='complete numerical positive rate and all-request median cost',
            enter='at least one match has rate_difference>=0 and median_cost_moe<median_cost_dense',
            intervals='descriptive paired input-block bootstrap; not an additional decision gate'),
        bootstrap=dict(resamples=10000,seed=20261029,unit='input; keep all three paired seeds together'),
        secondary_radii=[8], clean_correct_filter=False, statuses=['POSITIVE','UNSAFE_REPLAYED','UNKNOWN','TIMEOUT','ERROR'],
        source_complete_proof=False, controls=controls, freeze_preparation_seconds=time.monotonic()-began))
    print('FROZEN',path,'complete4800/incomplete24000/auxiliary24000')


def resources(workers):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields=line.split()
        if fields[0] in ('MemAvailable:', 'MemTotal:'):
            memory[fields[0][:-1]]=int(fields[1])*1024
    stat=os.statvfs('/data1/Kane/MOE')
    return dict(clock=time.time(),allowed_cpus=len(os.sched_getaffinity(0)),load=os.getloadavg(),
        memory=memory,ordinary_available_disk_bytes=stat.f_bavail*stat.f_frsize,workers=workers,
        gpu_used=False,action='read only; no signal or modification of another task')


def run(root,manifest):
    began=time.monotonic()
    require(ROOT != Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').is_file(), 'immutable study execution')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    spec=load(manifest)
    require(all(sha(ROOT/p)==h for p,h in spec['source_files'].items()),'frozen numerical code')
    require(digest(backend.source_identity())==spec['backend_sources_sha256'],'frozen numerical backend')
    for model in spec['models']:
        require(sha(model['checkpoint'])==model['checkpoint_sha256'],'frozen selected model')
    requests=all_requests(spec['models'])
    require(digest(requests)==spec['request_catalog_sha256'],'finite full queue identity')
    observation=resources(spec['workers'])
    require(observation['ordinary_available_disk_bytes']>=2*2**30,'at least2GiB free disk at admission')
    require(observation['load'][0]+spec['workers']<=observation['allowed_cpus'], 'CPU admission without overbooking')
    require(observation['memory']['MemAvailable']>=(8*spec['workers']+4)*2**30,'conservative process-cap memory headroom')
    store=Artifacts.create(root,initial_bytes=spec['inputs']['initial_account_bytes']+Path(manifest).stat().st_size,
        cap=spec['artifact_bytes'])
    store.write(store.root/'start.json.gz',dict(commit=(ROOT/'.frozen_commit').read_text().strip(),
        manifest=str(manifest),manifest_sha256=sha(manifest),resources=observation,
        preparation_seconds=time.monotonic()-began,pid=os.getpid()))
    summary=finite_queue(store.root,requests,manifest,workers=spec['workers'],seconds=300,
        timing_reserve=tuple(spec['timing_reserve']))
    print(summary)
    return 0 if summary['status']=='COMPLETE' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze(args.freeze)
    elif args.run and args.manifest:raise SystemExit(run(args.run,args.manifest))
    else:parser.error('use --freeze or --run with --manifest')
