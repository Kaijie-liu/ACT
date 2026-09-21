"""Two real native epochs (one batch each) and fresh-process continuation.

Explicit RNG checkpoint callback supplements Lightning's optimizer/scheduler
restore. Native model, augmentation, PGD7 and update remain unchanged. CPU only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from recent_moe_deployment import sha256, supervise
from robust_experts_workflow_control import compose, write


def rng_state():
    import random
    import numpy as np
    import torch
    n = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'numpy': {'kind': n[0], 'keys': n[1].tolist(), 'pos': n[2],
                      'has_gauss': n[3], 'cached': n[4]}}


def restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    n = state['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32),
                         n['pos'], n['has_gauss'], n['cached']))


def validate(cfg):
    from recent_moe_env_inventory import inventory
    for name, h in cfg['files'].items():
        if sha256(name) != h:
            raise ValueError('source/data identity: ' + name)
    if inventory([cfg['python']]) != json.loads(Path(cfg['environment']).read_text()):
        raise ValueError('environment identity')
    if cfg['epochs'] != 2 or cfg['total_seconds'] != 600 or cfg['workers'] != 0:
        raise ValueError('frozen resume control altered')


def worker(cfg, path, phase):
    start = time.monotonic()
    validate(cfg)
    import torch
    import pytorch_lightning as pl
    from omegaconf import open_dict
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    root = Path(cfg['output_root']) / phase
    root.mkdir()
    native, conf, _ = compose(Path(cfg['repo']), root, Path(cfg['data_root']))
    with open_dict(conf):
        conf.trainer.max_epochs = 2
        conf.execute_test = conf.execute_attack = False
        conf.callbacks.checkpoint.update({'save_top_k': -1, 'save_last': True,
            'filename': 'epoch{epoch:02d}', 'auto_insert_metric_name': False,
            'save_on_train_epoch_end': True})
    _, _, dm, model, trainer = native._setup(conf)
    dm.data_dir = cfg['data_root']  # same path-only author subclass correction
    binding = sha256(path)
    class StateAndBatch(pl.Callback):
        pending_rng = None
        def on_save_checkpoint(self, trainer, module, checkpoint):
            checkpoint['reproduction_rng'] = rng_state()
            checkpoint['reproduction_binding'] = binding
        def on_load_checkpoint(self, trainer, module, checkpoint):
            if checkpoint.get('reproduction_binding') != binding:
                raise ValueError('resume checkpoint binding')
            self.pending_rng = checkpoint['reproduction_rng']
        def on_train_start(self, trainer, module):
            if phase == 'resume':
                if self.pending_rng is None:
                    raise ValueError('resume RNG missing')
                restore_rng(self.pending_rng)
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            torch.save({'batch': batch, 'rng': rng_state()}, root / f'batch_epoch{trainer.current_epoch}.pt')
        def on_before_optimizer_step(self, trainer, module, optimizer, optimizer_idx):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('nonfinite/empty native gradients')
    trainer.callbacks.insert(0, StateAndBatch())
    checkpoint = None
    if phase == 'resume':
        reference = Path(cfg['output_root']) / 'reference'
        info = json.loads((reference / 'result.json').read_text())
        checkpoint = str(reference / 'checkpoints/epoch00.ckpt')
        if sha256(checkpoint) != info['checkpoint_hashes'][checkpoint]:
            raise ValueError('resume source changed')
    trainer.fit(model=model, datamodule=dm, ckpt_path=checkpoint)
    if trainer.global_step != 2:
        raise ValueError('native training cursor')
    final = root / 'checkpoints/epoch01.ckpt'
    if not final.exists():
        raise ValueError('completed-epoch checkpoint missing')
    write(root / 'result.json', {'status': 'COMPLETED', 'phase': phase,
        'config_sha256': binding, 'global_step': trainer.global_step,
        'checkpoint_hashes': {str(p): sha256(p) for p in sorted((root / 'checkpoints').glob('*.ckpt'))},
        'batch_hashes': {str(p): sha256(p) for p in root.glob('batch_epoch*.pt')},
        'worker_seconds': time.monotonic()-start})


def audit(cfg, path):
    import torch
    from audit_dual_rs_training_control import canonical
    start = time.monotonic()
    states, batches = {}, {}
    for phase in ['reference', 'resume']:
        root = Path(cfg['output_root']) / phase
        info = json.loads((root / 'result.json').read_text())
        if info['status'] != 'COMPLETED' or info['config_sha256'] != sha256(path):
            raise ValueError('phase incomplete/binding')
        for file, h in {**info['checkpoint_hashes'], **info['batch_hashes']}.items():
            if sha256(file) != h:
                raise ValueError('saved state changed')
        states[phase] = torch.load(root / 'checkpoints/epoch01.ckpt', weights_only=False, map_location='cpu')
        batches[phase] = torch.load(root / 'batch_epoch1.pt', weights_only=True)
    fields = ['state_dict', 'optimizer_states', 'lr_schedulers', 'global_step', 'epoch',
              'reproduction_rng', 'reproduction_binding']
    equal = {k: canonical(states['reference'][k]) == canonical(states['resume'][k]) for k in fields}
    equal['next_augmented_batch_and_rng'] = canonical(batches['reference']) == canonical(batches['resume'])
    accepted = all(equal.values()) and states['reference']['global_step'] == 2
    result = {'audit': 'EXACT_CONTINUATION_PASS' if accepted else 'CONTINUATION_DIFFERENCE',
              'accepted': accepted, 'equal': equal, 'config_sha256': sha256(path),
              'seconds': time.monotonic()-start,
              'scope': 'two CPU epochs, one native batch each, explicit RNG callback; NOT long training or GPU equivalence'}
    write(Path(cfg['output_root']) / 'audit.json', result)
    if not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--phase', choices=['manager', 'reference', 'resume', 'audit'])
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    root = Path(cfg['output_root'])
    if a.phase in ['reference', 'resume']:
        worker(cfg, a.config, a.phase)
    elif a.phase == 'audit':
        audit(cfg, a.config)
    elif a.phase == 'manager':
        for phase in ['reference', 'resume', 'audit']:
            began = time.monotonic()
            with (root / f'{phase}.stdout').open('x') as out, (root / f'{phase}.stderr').open('x') as err:
                code = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--config', str(a.config.resolve()),
                                       '--phase', phase], stdout=out, stderr=err).returncode
            write(root / f'{phase}_finished.json', {'returncode': code, 'seconds': time.monotonic()-began})
            if code:
                raise SystemExit(code)
        write(root / 'inner_terminal.json', {'status': 'EXACT_CONTINUATION_PASS'})
    else:
        receipt = supervise([cfg['python'], str(Path(__file__).resolve()), '--config', str(a.config.resolve()),
                             '--phase', 'manager'], str(Path(__file__).resolve().parents[1]), root,
                             cfg['total_seconds'], 'NATIVE_FULL_STATE_CONTINUATION', cfg['original_repo'])
        inner = root / 'inner_terminal.json'
        accepted = receipt['status'] == 'COMPLETED' and inner.exists()
        write(root / 'terminal.json', {'status': 'EXACT_CONTINUATION_PASS' if accepted else
            ('ERROR' if receipt['status'] == 'COMPLETED' else receipt['status']),
            'accepted': accepted, 'execution_seconds': receipt['execution_including_preflight_seconds'],
            'with_postflight_seconds': receipt['total_with_postflight_seconds']})
