"""Native train/test epoch supervision; explicit log-domain compatibility mode.

Only completed-epoch snapshots are resumable. Pending epoch work is retained,
never promoted or silently retried. Training selection is final epoch, not a
certification-selected checkpoint. Existing R1/R2 controls stay immutable.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import dual_rs_training_control as base
from dual_rs_training_control_r2 import enable_compatibility
from dual_rs_training_state import cpu_tree, digest, restore_rng, rng_state, save_snapshot
from recent_moe_deployment import sha256, supervise


def validate_protocol(cfg):
    if cfg['mode'] not in {'epoch_control', 'training'} or cfg['epochs'] != (2 if cfg['mode'] == 'epoch_control' else 90):
        raise ValueError('unregistered mode/epoch count')
    if sha256(cfg['parent_config']) != cfg['parent_config_sha256']:
        raise ValueError('parent identity')
    parent = json.loads(Path(cfg['parent_config']).read_text())
    base.validate_config(parent)
    for p, h in cfg['execution_files'].items():
        if sha256(p) != h:
            raise ValueError('epoch runner identity')
    archive = json.loads(Path(cfg['step_control_archive']).read_text())
    if archive['result'] != 'CONTROL_PASS' or sha256(cfg['step_control_archive']) != cfg['step_control_archive_sha256']:
        raise ValueError('step/restore prerequisite failed')
    if cfg['mode'] == 'training':
        epoch_archive = json.loads(Path(cfg['epoch_control_archive']).read_text())
        if (sha256(cfg['epoch_control_archive']) != cfg['epoch_control_archive_sha256'] or
                epoch_archive['audit'] != 'PASS' or not epoch_archive['exact_epoch_continuation']):
            raise ValueError('completed epoch control prerequisite')
    return parent


def check_epoch_state(state, binding, epoch, updates):
    import torch
    required = {'schema', 'binding', 'model', 'optimizer', 'scheduler', 'rng', 'completed_epoch', 'global_step'}
    if set(state) != required or state['schema'] != 'dual_rs_completed_epoch_v1' or state['binding'] != binding:
        raise ValueError('epoch state identity')
    if set(state['rng']) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('incomplete RNG state')
    if state['completed_epoch'] != epoch or state['global_step'] != updates:
        raise ValueError('incomplete epoch or wrong update count')
    if state['scheduler']['last_epoch'] != epoch:
        raise ValueError('scheduler not at completed epoch')
    opt = state['optimizer']['state']
    if not opt or any(float(x['step']) != updates for x in opt.values()):
        raise ValueError('AdamW update count')
    def finite(x):
        if isinstance(x, torch.Tensor):
            return bool(torch.isfinite(x).all())
        if isinstance(x, dict):
            return all(finite(v) for v in x.values())
        if isinstance(x, (tuple, list)):
            return all(finite(v) for v in x)
        return True
    if not finite(state):
        raise ValueError('nonfinite saved state')


def worker(cfg, config_path, phase):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    start = time.monotonic()
    parent = validate_protocol(cfg)
    enable_compatibility()
    native, model, optimizer, scheduler, denoiser, timestep, weights = base.build(parent)
    frozen_denoiser = digest(denoiser.state_dict())
    table_test = cfg['test_label_table']
    indices = {}
    for split, path in [('train', parent['label_table']), ('test', table_test)]:
        eligible = np.flatnonzero(np.load(path, allow_pickle=False)[:, -2] != 0).tolist()
        count = cfg['prefix_count'] if cfg['mode'] == 'epoch_control' else len(eligible)
        indices[split] = eligible[:count]
        if digest(indices[split]) != cfg[f'{split}_indices_sha256']:
            raise ValueError('dataset selection mismatch')
    datasets = {
        'train': native.get_dataset('cifar10', 'train_sigma_est', parent['label_table']),
        'test': native.get_dataset('cifar10', 'test_sigma_est', table_test)}
    loaders = {s: DataLoader(Subset(datasets[s], indices[s]), batch_size=256,
                shuffle=(s == 'train'), num_workers=0, pin_memory=False, drop_last=False) for s in datasets}
    # Retain author's two chunks per loader batch; these registered sizes are even.
    if len(indices['train']) % 256 % 2:
        raise ValueError('odd tail would lose an author chunk element; new protocol required')
    updates_per_epoch = 2 * math.ceil(len(indices['train']) / 256)
    binding = {'config_sha256': sha256(config_path), 'parent_config_sha256': cfg['parent_config_sha256'],
        'train_indices_sha256': cfg['train_indices_sha256'], 'test_indices_sha256': cfg['test_indices_sha256'],
        'denoiser_sha256': frozen_denoiser, 'compatibility': parent['consistency_implementation']}
    root = Path(cfg['output_root']) / phase
    root.mkdir(exist_ok=False)
    completed, updates = 0, 0
    if phase == 'resume':
        info = json.loads((Path(cfg['output_root']) / 'reference/epoch001.json').read_text())
        path = Path(cfg['output_root']) / 'reference/epoch001.pt'
        if sha256(path) != info['file_sha256']:
            raise ValueError('resume file identity')
        state = torch.load(path, map_location='cpu', weights_only=True)
        check_epoch_state(state, binding, 1, updates_per_epoch)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])  # last, after all constructors
        completed, updates = 1, updates_per_epoch

    def pre(_opt, _args, _kwargs):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError('nonfinite gradients; rejected before AdamW')

    def post(_opt, _args, _kwargs):
        nonlocal updates
        updates += 1
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite updated parameters')

    h1, h2 = optimizer.register_step_pre_hook(pre), optimizer.register_step_post_hook(post)
    try:
        for epoch in range(completed, cfg['epochs']):
            epoch_start = time.monotonic()
            base.write_json(root / f'epoch{epoch + 1:03d}_started.json', {
                'completed_epoch_before': epoch, 'global_step_before': updates, 'monotonic': epoch_start})
            train = native.train(loaders['train'], denoiser, timestep, model, optimizer, epoch,
                                 1.0, torch.device('cuda'), writer=None, class_weights=weights)
            after_train = time.monotonic()
            test = native.test(loaders['test'], denoiser, timestep, model, epoch, 1.0,
                               torch.device('cuda'), native.args, writer=None, num_noise_vec=2)
            if not all(math.isfinite(float(v)) for v in [*train, *test]):
                raise ValueError('nonfinite epoch metrics')
            after_test = time.monotonic()
            scheduler.step()  # author order: train, test, scheduler, save
            state = cpu_tree({'schema': 'dual_rs_completed_epoch_v1', 'binding': binding,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state(),
                'completed_epoch': epoch + 1, 'global_step': updates})
            check_epoch_state(state, binding, epoch + 1, (epoch + 1) * updates_per_epoch)
            identity = save_snapshot(root / f'epoch{epoch + 1:03d}.pt', state)
            completed = epoch + 1
            base.write_json(root / f'epoch{completed:03d}.json', {
                **identity, 'binding': binding, 'completed_epoch': completed, 'global_step': updates,
                'train': [float(v) for v in train], 'test_author_metrics': [float(v) for v in test],
                'lr_next_epoch': scheduler.get_last_lr(), 'train_seconds': after_train - epoch_start,
                'test_seconds': after_test - after_train, 'save_seconds': time.monotonic() - after_test,
                'epoch_total_seconds': time.monotonic() - epoch_start})
        if digest(denoiser.state_dict()) != frozen_denoiser:
            raise ValueError('frozen denoiser changed')
        base.write_json(root / 'terminal.json', {'status': 'COMPLETED', 'completed_epochs': completed,
            'global_step': updates, 'binding': binding, 'worker_seconds': time.monotonic() - start,
            'denoiser_unchanged': True, 'selected_checkpoint': f'epoch{completed:03d}.pt',
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    except BaseException as exc:
        base.write_json(root / 'terminal.json', {'status': 'ERROR', 'error': repr(exc),
            'completed_epochs': completed, 'partial_global_step': updates,
            'worker_seconds': time.monotonic() - start})
        raise
    finally:
        h1.remove()
        h2.remove()


def summarize_terminal(root, receipt):
    value = json.loads((root / 'terminal.json').read_text()) if (root / 'terminal.json').exists() else {}
    if receipt['status'] != 'COMPLETED':
        value = {'status': receipt['status'], 'partial_evidence_preserved': True}
    elif value.get('status') not in {'CONTROL_PASS', 'TRAINING_LANDED'}:
        value = {'status': 'ERROR', 'reason': 'missing successful inner terminal'}
    value['execution_seconds'] = receipt['execution_including_preflight_seconds']
    value['with_postflight_seconds'] = receipt['total_with_postflight_seconds']
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--phase', choices=['reference', 'resume', 'training', 'manager'])
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    root = Path(cfg['output_root'])
    if args.phase in {'reference', 'resume', 'training'}:
        worker(cfg, args.config, args.phase)
        return
    if args.phase == 'manager':
        phases = ['reference', 'resume', 'audit'] if cfg['mode'] == 'epoch_control' else ['training', 'audit']
        for phase in phases:
            base.write_json(root / f'{phase}_started.json', {'phase': phase, 'monotonic': time.monotonic()})
            command = ([sys.executable, str(Path(__file__).resolve()), '--config', str(args.config.resolve()), '--phase', phase]
                if phase != 'audit' else [sys.executable, str(Path(__file__).with_name('audit_dual_rs_epochs.py')),
                       '--config', str(args.config.resolve()), '--output', str(root / 'audit.json')])
            start = time.monotonic()
            with (root / f'{phase}.stdout').open('x') as out, (root / f'{phase}.stderr').open('x') as err:
                code = subprocess.run(command, stdout=out, stderr=err).returncode
            base.write_json(root / f'{phase}_finished.json', {'phase': phase, 'returncode': code,
                            'wall_seconds': time.monotonic() - start})
            if code:
                raise SystemExit(code)
        base.write_json(root / 'terminal.json', {'status': 'CONTROL_PASS' if cfg['mode'] == 'epoch_control'
                         else 'TRAINING_LANDED', 'epochs': cfg['epochs']})
        return
    parent = json.loads(Path(cfg['parent_config']).read_text())
    receipt = supervise([parent['python'], str(Path(__file__).resolve()), '--config', str(args.config.resolve()),
        '--phase', 'manager'], str(Path(__file__).resolve().parents[1]), root, cfg['total_seconds'],
        'DUAL_RS_COMPLETED_EPOCH_PIPELINE', parent['author_repo'], cpu_only=False)
    value = summarize_terminal(root, receipt)
    base.write_json(root / 'outer_terminal.json', value)
    print(json.dumps(value, indent=2))
    raise SystemExit(0 if value.get('status') in {'CONTROL_PASS', 'TRAINING_LANDED'} else 1)


if __name__ == '__main__':
    main()
