"""Frozen 24-run CPU S2a training. No test split is evaluated here.

This is a training queue, not a new verifier supervisor. It reuses the existing
atomic checkpoint writer, process resource setup and shared artifact account.
"""
import time
ENTRY = time.monotonic()
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha
from multilayer_train import setup, atomic_torch_save
from architecture_s1_execution import Artifacts, THREAD_ENV


def verify_source(manifest):
    spec = load(manifest)
    if ROOT == Path('/data1/Kane/MOE/ACT') or not (ROOT/'.frozen_commit').is_file():
        raise ValueError('requires immutable committed execution copy')
    for p, digest in spec['source_files'].items():
        if sha(ROOT/p) != digest:
            raise ValueError('frozen S2 source '+p)
    if sha(spec['control_log']) != spec['control_log_sha256']:
        raise ValueError('bound synthetic controls changed')
    return spec


def train_one(folder, manifest, item_path):
    spec = verify_source(manifest); item = load(item_path)
    store = Artifacts(folder); target = folder/item['id']
    if (target/'complete.json').exists():
        raise ValueError('completed training must not be rerun')
    target.mkdir(exist_ok=True)
    torch = setup(1)
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from act.pipeline.moe.architecture_s2_training import (
        build, schedule, objective, evaluate_ibp, selection_key,
    )
    for p, digest in spec['training_data_files'].items():
        if sha(p) != digest:
            raise ValueError('frozen training data '+p)
    dataset = CIFAR10(spec['dataset_root'], train=True, download=False, transform=ToTensor())
    train_data, validation_data = random_split(dataset, [45000, 5000],
        generator=torch.Generator().manual_seed(spec['split_seed']))
    split = dict(train=train_data.indices, validation=validation_data.indices)
    # Match the already published split byte convention; no new sampling.
    if not (folder/'split.json').exists():
        raise ValueError('controller has not frozen the split')
    if load(folder/'split.json') != split or sha(folder/'split.json') != spec['split_sha256']:
        raise ValueError('training/validation split mismatch')
    torch.manual_seed(item['seed']); model = build(item)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != item['parameter_count']:
        raise ValueError('frozen parameter count')
    generator = torch.Generator().manual_seed(item['seed'])
    loader = DataLoader(train_data, batch_size=spec['batch_size'], shuffle=True,
                        num_workers=0, generator=generator)
    validation = DataLoader(validation_data, batch_size=spec['batch_size'],
                            shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=item['lr'],
                                  weight_decay=spec['weight_decay'])
    first_epoch = 1; best_state = best_metrics = best_epoch = None
    resumed_seconds = 0.
    resume = target/'resume.pt'
    if resume.exists():
        old = torch.load(resume, map_location='cpu', weights_only=True)
        if old['manifest_sha256'] != sha(manifest) or old['item'] != item:
            raise ValueError('resume source/configuration identity')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        torch.set_rng_state(old['rng']); generator.set_state(old['shuffle_rng'])
        best_state, best_metrics, best_epoch = old['best_state'], old['best_metrics'], old['best_epoch']
        first_epoch = old['epoch']+1; resumed_seconds = old['seconds']
        missing = target/f'epoch_{old["epoch"]:03d}.json'
        if not missing.exists(): store.write(missing, old['last_epoch_record'])
    for epoch in range(first_epoch, spec['epochs']+1):
        started = time.monotonic(); model.train()
        sums = dict(loss=0., natural=0., robust=0., route_penalty=0., balance=0., correct=0)
        total = 0
        for batch, (x, y) in enumerate(loader):
            eps, kappa = schedule(epoch, batch, len(loader), spec)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(model, x, y, epsilon=eps, kappa=kappa,
                route_weight=item['route_weight'], family=item['family'])
            loss.backward(); optimizer.step()
            sums['loss'] += float(loss.detach())*len(y)
            for key, value in metrics.items():
                sums[key] += value if key == 'correct' else value*len(y)
            total += len(y)
        training_seconds = time.monotonic()-started
        measured = evaluate_ibp(model, validation, spec['epsilon'])
        if best_metrics is None or selection_key(measured) > selection_key(best_metrics):
            best_metrics = measured; best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        row = dict(epoch=epoch, total=total, metrics={k:v/total for k,v in sums.items()},
            epsilon_last=eps, kappa_last=kappa, validation=measured,
            train_seconds=training_seconds, validation_seconds=time.monotonic()-started-training_seconds,
            seconds=time.monotonic()-started, best_epoch=best_epoch)
        # Resume saved first: any terminal interruption preserves the exact next
        # epoch RNG/optimizer. Existing epoch records are never replaced.
        atomic_torch_save(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            best_state=best_state, best_epoch=best_epoch, best_metrics=best_metrics, epoch=epoch,
            rng=torch.get_rng_state(), shuffle_rng=generator.get_state(), item=item,
            manifest_sha256=sha(manifest), seconds=resumed_seconds+time.monotonic()-ENTRY,
            last_epoch_record=row), torch)
        store.write(target/f'epoch_{epoch:03d}.json', row)
        store.write(target/'progress.json.gz', dict(epoch=epoch, best_epoch=best_epoch,
            validation=measured, seconds=resumed_seconds+time.monotonic()-ENTRY), replace=True)
    # If interrupted after resume publication, recover its bound epoch record;
    # this is training progress, never a retroactive verifier success.
    if resume.exists():
        old = torch.load(resume, map_location='cpu', weights_only=True)
        missing = target/f'epoch_{old["epoch"]:03d}.json'
        if not missing.exists(): store.write(missing, old['last_epoch_record'])
    if best_state is None or best_metrics is None:
        raise ValueError('no completed training checkpoint')
    model.load_state_dict(best_state)
    checkpoint = target/'selected.pt'
    payload = dict(format='act-dense-capacity-s1-v1' if item['family']=='dense' else 'act-sequential-top2-v1',
        architecture=item['architecture'], seed=item['seed'], state_dict=best_state,
        parameter_count=parameter_count, best_epoch=best_epoch, validation=best_metrics,
        split_sha256=spec['split_sha256'], manifest_sha256=sha(manifest),
        code_commit=(ROOT/'.frozen_commit').read_text().strip(), training_method='S2a_route_envelope_IBP',
        item=item, test_evaluated=False)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['manifest_sha256'] != sha(manifest) or saved['item'] != item or any(
                not torch.equal(saved['state_dict'][k], v) for k, v in best_state.items()):
            raise ValueError('existing selected checkpoint differs')
    else:
        atomic_torch_save(checkpoint, payload, torch)
    result = dict(item=item, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size, resume_bytes=resume.stat().st_size,
        best_epoch=best_epoch, validation=best_metrics, test_evaluated=False,
        seconds=resumed_seconds+time.monotonic()-ENTRY)
    store.write(target/'complete.json', result, terminal=True)


def run_queue(folder, manifest, workers):
    spec = verify_source(manifest)
    if not 1 <= workers <= spec['max_workers']:
        raise ValueError('worker count')
    if not folder.exists():
        store = Artifacts.create(folder, initial_bytes=spec['control_artifact_bytes'],
                                 cap=spec['artifact_cap_bytes'])
        # Reuse exact historical split bytes (no test data or new RNG dependency).
        split = Path(spec['split_source'])
        if sha(split) != spec['split_sha256']:
            raise ValueError('frozen split source')
        (folder/'split.json').write_bytes(split.read_bytes())
        store.write(folder/'split_binding.json', dict(bytes=split.stat().st_size,
                    sha256=sha(split)))
        # Split was precharged in control_artifact_bytes by the freeze.
        store.write(folder/'start.json', dict(manifest_sha256=sha(manifest), workers=workers,
            pid=os.getpid(), started_at=time.time(), immutable_commit=(ROOT/'.frozen_commit').read_text().strip()))
    else:
        store = Artifacts(folder)
        if load(folder/'start.json')['manifest_sha256'] != sha(manifest):
            raise ValueError('queue resume manifest')
    from act.pipeline.moe.architecture_s2_training import tune_items, selected_followups
    items = tune_items(spec)

    def execute(item):
        target = folder/item['id']; target.mkdir(exist_ok=True)
        item_path = target/'item.json'
        if item_path.exists():
            if load(item_path) != item: raise ValueError('queue item identity')
        else:
            store.write(item_path, item)
        if (target/'complete.json').exists():
            value = load(target/'complete.json')
            if value['item'] != item or sha(value['checkpoint']) != value['checkpoint_sha256']:
                raise ValueError('completed training identity')
            return value
        attempt = len(list(target.glob('attempt_*.json')))
        began = time.monotonic()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(item_path),
            '--manifest', str(manifest), '--folder', str(folder)],
            env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True, start_new_session=True)
        store.write(target/f'attempt_{attempt:03d}.json', dict(returncode=child.returncode,
            seconds=time.monotonic()-began, stdout=child.stdout, stderr=child.stderr), terminal=True)
        if child.returncode != 0:
            raise RuntimeError(f'training {item["id"]} returned {child.returncode}; prefix retained')
        value = load(target/'complete.json')
        if value['item'] != item or sha(value['checkpoint']) != value['checkpoint_sha256']:
            raise ValueError('training result identity')
        return value

    def phase(roster):
        iterator = iter(roster); results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(execute, item) for item in list_next(iterator, workers)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                # Check every finished result BEFORE launching another training.
                results.extend(f.result() for f in done)
                for item in list_next(iterator, len(done)):
                    pending.add(pool.submit(execute, item))
        return results

    tuned = phase(items)
    selected, followups = selected_followups(spec, tuned)
    if len(items)+len(followups) != spec['max_training_runs']:
        raise ValueError('24-run roster')
    selection = dict(configurations={k:dict(id=v['item']['id'], configuration=v['item']['configuration'],
        validation=v['validation'], checkpoint_sha256=v['checkpoint_sha256']) for k,v in selected.items()},
        seed0_reused=True, test_evaluated=False)
    if (folder/'selection.json').exists():
        if load(folder/'selection.json') != selection: raise ValueError('selection changed')
    else: store.write(folder/'selection.json', selection)
    trained = phase(followups)
    final = [selected[a['id']] for a in spec['arms']] + trained
    if len(final) != 20: raise ValueError('twenty selected models')
    store.write(folder/'complete.json', dict(status='TRAINING_COMPLETE', models=final,
        training_runs=24, selected_models=20, test_evaluated=False,
        controller_seconds=time.monotonic()-ENTRY,
        sum_attempt_seconds=sum(load(p)['seconds'] for p in folder.glob('*/attempt_*.json')),
        model_storage_bytes=sum(p.stat().st_size for p in folder.rglob('*.pt')),
        evidence_account=load(store.account)), terminal=True)


def list_next(iterator, count):
    import itertools
    return list(itertools.islice(iterator, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    try:
        if args.worker:
            train_one(args.folder, args.manifest, args.worker)
        else:
            os.nice(10)
            # Controller also imports torch to build only the catalogue routines;
            # no model is constructed and no data split is loaded in this process.
            setup(1)
            run_queue(args.folder, args.manifest, args.workers)
    except BaseException as exc:
        if args.folder.exists():
            target = args.folder/(args.worker.parent.name if args.worker else '')
            num = len(list(target.glob('error_*.json')))
            Artifacts(args.folder).write(target/f'error_{num:03d}.json', dict(
                kind=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                seconds=time.monotonic()-ENTRY), terminal=True)
        raise
