"""Independent saved epoch accounting and exact continuation comparison."""
import argparse
import json
from pathlib import Path
import math
import time

from audit_dual_rs_training_control import canonical
from recent_moe_deployment import sha256


def audit(config):
    import torch
    start = time.monotonic()
    cfg = json.loads(config.read_text())
    root = Path(cfg['output_root'])
    phases = ['reference', 'resume'] if cfg['mode'] == 'epoch_control' else ['training']
    states, epoch_metadata = {}, {}
    updates_per_epoch = 2 * math.ceil(cfg['train_count'] / 256)
    for phase in phases:
        path = root / phase
        terminal = json.loads((path / 'terminal.json').read_text())
        if terminal['status'] != 'COMPLETED' or not terminal['denoiser_unchanged'] or terminal['completed_epochs'] != cfg['epochs']:
            raise ValueError('not completed')
        epoch_metadata[phase] = []
        for epoch in range(2 if phase == 'resume' else 1, cfg['epochs'] + 1):
            info = json.loads((path / f'epoch{epoch:03d}.json').read_text())
            file = path / f'epoch{epoch:03d}.pt'
            if sha256(file) != info['file_sha256'] or info['binding']['config_sha256'] != sha256(config):
                raise ValueError('epoch file binding')
            state = torch.load(file, map_location='cpu', weights_only=True)
            if (state['binding'] != info['binding'] or state['completed_epoch'] != epoch or
                    state['global_step'] != updates_per_epoch * epoch or
                    state['scheduler']['last_epoch'] != epoch):
                raise ValueError('state/cursor/scheduler mismatch')
            if not state['optimizer']['state'] or any(float(m['step']) != updates_per_epoch * epoch
                                            for m in state['optimizer']['state'].values()):
                raise ValueError('AdamW step count')
            for group in ['model', 'optimizer']:
                def check(v):
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                        raise ValueError('nonfinite training state')
                    if isinstance(v, dict):
                        for x in v.values():
                            check(x)
                    if isinstance(v, (list, tuple)):
                        for x in v:
                            check(x)
                check(state[group])
            if any(info[k] < 0 for k in ['train_seconds', 'test_seconds', 'save_seconds', 'epoch_total_seconds']):
                raise ValueError('negative cost')
            if info['epoch_total_seconds'] + 1e-6 < sum(info[k] for k in ['train_seconds', 'test_seconds', 'save_seconds']):
                raise ValueError('incomplete cost')
            epoch_metadata[phase].append(info)
        states[phase] = state
    exact = None
    if cfg['mode'] == 'epoch_control':
        exact = canonical(states['reference']) == canonical(states['resume'])
        if not exact:
            raise ValueError('fresh-process epoch continuation differs')
        for key in ['train', 'test_author_metrics', 'lr_next_epoch', 'global_step']:
            if epoch_metadata['reference'][-1][key] != epoch_metadata['resume'][-1][key]:
                raise ValueError('epoch diagnostics differ')
    return {'audit': 'PASS', 'config_sha256': sha256(config), 'mode': cfg['mode'],
        'epochs': cfg['epochs'], 'exact_epoch_continuation': exact,
        'checkpoint_selection': 'last registered epoch; no certification/accuracy selection',
        'epoch_records': epoch_metadata, 'audit_seconds': time.monotonic() - start,
        'scope': 'saved-state and accounting audit; no claim of final classifier certification'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.config)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print('Saved epoch accounting PASS')
