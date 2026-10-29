"""Frozen S1 dense controls; training time is separate from verification time."""
import time
ENTRY = time.monotonic()
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha
from multilayer_train import setup, atomic_torch_save


def control(folder):
    folder.mkdir()
    torch = setup(1)
    from act.pipeline.moe.architecture_dense import build, load_checkpoint, parameter_count, matched_widths
    from act.pipeline.moe.multilayer_training import build as moe_build
    old = load(ROOT/'docs/multilayer_training_freeze_20261022.json')
    moe = moe_build(old['models'][0]['architecture'], 0)
    router = sum(p.numel() for p in moe.layers[0].router.parameters())
    expert = sum(p.numel() for p in moe.layers[0].experts[0].parameters())
    total = sum(p.numel() for p in moe.parameters())
    if (router, expert, total) != (394376, 820874, 6961368):
        raise ValueError('L1 capacity identity changed')
    rows = []
    for target in (total, router+2*expert):
        widths = matched_widths(target)
        model = build(dict(widths=widths), 0)
        assert sum(p.numel() for p in model.parameters()) == parameter_count(widths)
        y = model(torch.zeros(2, 3, 32, 32))
        assert y.shape == (2, 10) and torch.isfinite(y).all()
        rows.append(dict(target=target, widths=widths, parameters=parameter_count(widths)))
        del model
    config = dict(widths=[4, 6, 3, 2])
    model = build(config, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)/16
    y = torch.tensor([0, 1, 0, 1])
    before = [p.detach().clone() for p in model.parameters()]
    def step(net, opt):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(net(x), y)
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
    step(model, optimizer)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    shuffle = torch.Generator().manual_seed(4)
    torch.randperm(8, generator=shuffle)
    payload = dict(format='act-dense-capacity-s1-v1', architecture=config, seed=4,
        state_dict=copy.deepcopy(model.state_dict()), parameter_count=parameter_count(config['widths']),
        optimizer=copy.deepcopy(optimizer.state_dict()), rng=torch.get_rng_state(),
        shuffle_rng=shuffle.get_state())
    path = folder/'control.pt'
    atomic_torch_save(path, payload, torch)
    restored, restored_payload = load_checkpoint(path)
    assert torch.equal(model(x), restored(x))
    resumed = torch.optim.AdamW(restored.parameters(), lr=.001, weight_decay=.0001)
    resumed.load_state_dict(restored_payload['optimizer'])
    restored_shuffle = torch.Generator(); restored_shuffle.set_state(restored_payload['shuffle_rng'])
    assert torch.equal(torch.randperm(8, generator=shuffle), torch.randperm(8, generator=restored_shuffle))
    step(model, optimizer); step(restored, resumed)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    broken = dict(payload, parameter_count=payload['parameter_count']+1)
    atomic_torch_save(folder/'wrong_count.pt', broken, torch)
    try:
        load_checkpoint(folder/'wrong_count.pt')
    except ValueError:
        pass
    else:
        raise AssertionError('wrong parameter identity accepted')
    # Checkpoint choice is independent of certification and keeps earliest ties.
    best = -1.; selected = None
    for epoch, acc in enumerate([.4, .6, .6, .5], 1):
        if acc > best: best, selected = acc, epoch
    assert selected == 2
    save(folder/'complete.json', dict(status='PASS', capacities=rows,
        effective_step=True, output_restore=True, optimizer_resume=True,
        shuffle_resume=True, wrong_count_rejected=True, earliest_validation_tie=True,
        seconds=time.monotonic()-ENTRY, real_training=0, verification_calls=0))


def train(folder, manifest):
    spec = load(manifest)
    if ROOT == Path('/data1/Kane/MOE/ACT') or not (ROOT/'.frozen_commit').is_file():
        raise ValueError('training requires immutable committed execution copy')
    for p, digest in spec['source_files'].items():
        if sha(ROOT/p) != digest: raise ValueError('frozen training source '+p)
    if load(spec['control_result'], spec['control_sha256'])['status'] != 'PASS':
        raise ValueError('training controls not passed')
    receipt = load(spec['control_receipt'], spec['control_receipt_sha256'])
    if receipt['status'] != 'COMPLETED' or receipt['cleanup_status'] != 'LEADER_REAPED_NO_LIVE_SESSION':
        raise ValueError('training control supervision')
    folder.mkdir()
    torch = setup(1)
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from act.pipeline.moe.architecture_dense import build, parameter_count
    for p, digest in spec['data_files'].items():
        if sha(p) != digest: raise ValueError('fixed CIFAR data identity')
    dataset = CIFAR10(spec['dataset_root'], train=True, download=False, transform=ToTensor())
    train_data, val_data = random_split(dataset, [45000, 5000],
        generator=torch.Generator().manual_seed(spec['split_seed']))
    save(folder/'split.json', dict(train=train_data.indices, validation=val_data.indices))
    split_hash = sha(folder/'split.json')
    if split_hash != spec['split_sha256']: raise ValueError('old/common split differs')
    save(folder/'start.json', dict(commit=(ROOT/'.frozen_commit').read_text().strip(),
        manifest_sha256=sha(manifest), split_sha256=split_hash, pid=os.getpid(),
        started_at=time.time(), threads=1, training_seconds_separate=True))
    @torch.no_grad()
    def evaluate(model, loader):
        model.eval(); correct = total = 0
        for x, y in loader:
            correct += int((model(x).argmax(1) == y).sum()); total += len(y)
        return dict(accuracy=correct/total, correct=correct, total=total)
    summary = []
    for item in spec['models']:
        began = time.monotonic(); target = folder/item['id']; target.mkdir()
        torch.manual_seed(item['seed']); model = build(item['architecture'], item['seed'])
        count = sum(p.numel() for p in model.parameters())
        if count != item['parameter_count']: raise ValueError('capacity mismatch')
        loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(item['seed']))
        validation = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        best_accuracy = -1.; best_state = best_epoch = best_metrics = None
        for epoch in range(1, 51):
            epoch_start = time.monotonic(); model.train(); running = correct = total = 0
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True); value = model(x)
                loss = torch.nn.functional.cross_entropy(value, y)
                if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                loss.backward(); optimizer.step()
                running += float(loss.detach())*len(y)
                correct += int((value.detach().argmax(1) == y).sum()); total += len(y)
            metrics = evaluate(model, validation)
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']; best_epoch = epoch; best_metrics = metrics
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            row = dict(epoch=epoch, loss=running/total, train_accuracy=correct/total,
                validation=metrics, seconds=time.monotonic()-epoch_start,
                model_seconds=time.monotonic()-began)
            save(target/f'epoch_{epoch:03d}.json', row)
            atomic_torch_save(target/'resume.pt', dict(model=model.state_dict(),
                optimizer=optimizer.state_dict(), best_state=best_state,
                best_epoch=best_epoch, best_metrics=best_metrics, epoch=epoch,
                rng=torch.get_rng_state(), shuffle_rng=loader.generator.get_state(), item=item,
                manifest_sha256=sha(manifest), split_sha256=split_hash), torch)
            print(json.dumps(dict(model=item['id'], **row)), flush=True)
        model.load_state_dict(best_state)
        test = CIFAR10(spec['dataset_root'], train=False, download=False, transform=ToTensor())
        test_metrics = evaluate(model, DataLoader(test, batch_size=256, shuffle=False, num_workers=0))
        path = target/'selected.pt'
        atomic_torch_save(path, dict(format='act-dense-capacity-s1-v1', architecture=item['architecture'],
            seed=item['seed'], state_dict=best_state, parameter_count=count, best_epoch=best_epoch,
            validation=best_metrics, test=test_metrics, split_sha256=split_hash,
            manifest_sha256=sha(manifest), code_commit=(ROOT/'.frozen_commit').read_text().strip()), torch)
        result = dict(id=item['id'], parameter_count=count, best_epoch=best_epoch,
            validation=best_metrics, test=test_metrics, checkpoint=str(path),
            checkpoint_sha256=sha(path), seconds=time.monotonic()-began)
        save(target/'complete.json', result); summary.append(result)
    save(folder/'complete.json', dict(status='COMPLETE', models=summary,
        seconds=time.monotonic()-ENTRY, verification_calls=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--control', type=Path)
    p.add_argument('--run', type=Path); p.add_argument('--manifest', type=Path)
    args = p.parse_args()
    try:
        if args.control: control(args.control)
        elif args.run and args.manifest: train(args.run, args.manifest)
        else: p.error('use --control or --run with --manifest')
    except BaseException as exc:
        folder = args.control or args.run
        if folder and folder.exists():
            save(folder/'error.json', dict(kind=type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc(), seconds=time.monotonic()-ENTRY))
        raise
