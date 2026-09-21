"""Native train -> final-weight binding -> clean/PGD/APGD -> saved audit.

One outer budget per architecture includes ALL subprocess stages. No automatic
retry/resume, no best-checkpoint selection, no network or numerical fallback.
The short control is NOT a 200-epoch reproduction or a robustness estimate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from recent_moe_deployment import sha256, supervise
from recent_moe_env_inventory import inventory
from robust_experts_workflow_control import write

STAGES = ('train', 'evaluate', 'audit')


def configuration(recipe, root, data, mode):
    from omegaconf import OmegaConf, open_dict
    if mode not in ('control', 'training'):
        raise ValueError('unknown mode')
    cfg = OmegaConf.create(recipe)
    with open_dict(cfg):
        cfg.work_dir, cfg.data_dir = str(root), str(data)
        cfg.datamodule.data_dir = str(data)
        cfg.execute_train = cfg.execute_test = cfg.execute_attack = True
        cfg.use_clearml, cfg.wandb = False, {}
        cfg.optimized_metric, cfg.print_config = None, False
        cfg.logger = {'csv': {'_target_': 'pytorch_lightning.loggers.CSVLogger',
            'save_dir': str(root), 'name': 'local', 'version': 'pipeline'}}
        # Keep every completed epoch; no validation-based model choice or
        # overwriting the only recovery state. Partial failures are retained.
        cfg.callbacks = {'checkpoint': {'_target_': 'pytorch_lightning.callbacks.ModelCheckpoint',
            'dirpath': str(root/'checkpoints'), 'save_top_k': -1, 'save_last': False,
            'filename': 'epoch{epoch:03d}', 'auto_insert_metric_name': False,
            'every_n_epochs': 1, 'save_on_train_epoch_end': True}}
        cfg.trainer.min_epochs = cfg.trainer.max_epochs = 2 if mode == 'control' else 200
        cfg.model.scheduler.T_max = 200  # short control does not change LR horizon
        cfg.trainer.resume_from_checkpoint = None
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.enable_progress_bar = cfg.trainer.enable_model_summary = False
        cfg.trainer.default_root_dir = str(root)
        cfg.trainer.log_every_n_steps = 1
        if mode == 'control':
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.limit_test_batches = 1
    if (cfg.datamodule.batch_size != 640 or cfg.datamodule.num_workers != 2 or
            cfg.model.optimizer.lr != .01 or cfg.model.attack.steps != 7 or
            cfg.attack_batch_size != 256 or cfg.trainer.gpus != 1 or cfg.seed != 12345):
        raise ValueError('frozen scientific recipe changed')
    return cfg


def state_digest(state):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def check_binding(config, path):
    if config['mode'] not in ('control', 'training'):
        raise ValueError('mode')
    if config['mode'] == 'training' and not config.get('execution_freeze_approved'):
        raise ValueError('long training execution has not been frozen')
    for name, digest in config['files'].items():
        if sha256(name) != digest:
            raise ValueError('source/data drift: ' + name)
    if inventory([config['python']]) != config['environment']:
        raise ValueError('environment drift')
    if not Path(path).is_file():
        raise ValueError('missing configuration')


def native_setup(config, path, arm, stage):
    check_binding(config, path)
    from author_local_ipc import install
    install()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if stage != 'audit':
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 48*1024**3:
            raise ValueError('GPU_RESOURCE_WAIT_REQUIRED; no eviction/retry/fallback')
        torch.cuda.set_per_process_memory_fraction(.4, 0)
    sys.path.insert(0, config['repo'])
    from src import run as native
    root = Path(config['output_root'])/arm
    cfg = configuration(config['recipes'][arm], root, config['data_root'], config['mode'])
    return root, cfg, native


def train(config, path, arm):
    began = time.monotonic()
    root, cfg, native = native_setup(config, path, arm, 'train')
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    import torch
    from robust_experts_resume_control import rng_state
    if shutil.disk_usage(root).free < (5 if config['mode'] == 'control' else 128)*1024**3:
        raise ValueError('insufficient space for retained epoch checkpoints')
    write(root/'prepared.json', {'config_sha256': sha256(path), 'arm': arm,
        'symbolic_config': OmegaConf.to_container(cfg, resolve=False),
        'mode': config['mode'], 'automatic_resume': False})
    _, _, dm, model, trainer = native._setup(cfg)
    dm.data_dir = config['data_root']
    observations = {'updates': 0, 'epoch_batches': {}}
    class Journal(pl.Callback):
        def on_save_checkpoint(self, trainer, module, checkpoint):
            checkpoint['reproduction_binding'] = {'config_sha256': sha256(path), 'arm': arm}
            checkpoint['reproduction_rng'] = rng_state()
            checkpoint['reproduction_cuda_rng'] = torch.cuda.get_rng_state_all()
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if batch[0].device.type != 'cuda' or len(batch[0]) > 640:
                raise ValueError('native batch/device changed')
            observations['epoch_batches'].setdefault(str(trainer.current_epoch), []).append(len(batch[0]))
        def on_before_optimizer_step(self, trainer, module, optimizer, optimizer_idx):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('nonfinite native gradients')
            observations['updates'] += 1
        def on_train_epoch_end(self, trainer, module):
            write(root/f'epoch{trainer.current_epoch:03d}.json', {
                'epoch': trainer.current_epoch, 'global_step': trainer.global_step,
                'elapsed_seconds': time.monotonic()-began,
                'train_batch_sizes': observations['epoch_batches'][str(trainer.current_epoch)],
                'lr': [g['lr'] for g in trainer.optimizers[0].param_groups]})
    trainer.callbacks.insert(0, Journal())
    trainer.fit(model=model, datamodule=dm)
    epochs = 2 if config['mode'] == 'control' else 200
    if trainer.current_epoch != epochs or len(observations['epoch_batches']) != epochs:
        raise ValueError('incomplete epoch roster')
    if observations['updates'] != trainer.global_step or trainer.global_step < epochs:
        raise ValueError('update accounting')
    if config['mode'] == 'control' and any(v != [640] for v in observations['epoch_batches'].values()):
        raise ValueError('control requires two full640 batches')
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite final state')
    final = root/'final_epoch.ckpt'
    if final.exists():
        raise FileExistsError(final)
    trainer.save_checkpoint(final)
    torch.cuda.synchronize()
    write(root/'train.json', {'status': 'COMPLETED', 'config_sha256': sha256(path),
        'arm': arm, 'epochs': epochs, 'global_step': trainer.global_step,
        'checkpoint_sha256': sha256(final), 'state_digest': state_digest(model.state_dict()),
        'epoch_checkpoints': {p.name: sha256(p) for p in sorted((root/'checkpoints').glob('*.ckpt'))},
        'observations': observations, 'seconds': time.monotonic()-began,
        'selected_by': 'completed final epoch only', 'exact_gpu_resume': False})


def evaluate(config, path, arm):
    began = time.monotonic()
    root, cfg, native = native_setup(config, path, arm, 'evaluate')
    import torch
    import hydra
    import pytorch_lightning as pl
    final = root/'final_epoch.ckpt'
    trained = json.loads((root/'train.json').read_text())
    if trained['checkpoint_sha256'] != sha256(final) or trained['config_sha256'] != sha256(path):
        raise ValueError('final checkpoint binding')
    saved = torch.load(final, map_location='cpu', weights_only=False)
    attacks, _, dm, model, trainer = native._setup(cfg)
    dm.data_dir = config['data_root']
    model.load_state_dict(saved['state_dict'], strict=True)
    if state_digest(model.state_dict()) != trained['state_digest']:
        raise ValueError('evaluation model changed')
    if len(attacks) != 2 or any(a.model is not model.model for a in attacks):
        raise ValueError('attack does not share original full model')
    attack_dm = hydra.utils.instantiate(cfg.datamodule, batch_size=256, use_clearml=False)
    attack_dm.data_dir = config['data_root']
    counts = {'examples': 0, 'batches': 0}
    class Count(pl.Callback):
        def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
            counts['examples'] += len(batch[0])
            counts['batches'] += 1
    trainer.callbacks.append(Count())
    results = []
    for name, module, data in [('clean', model, dm), ('PGD20', attacks[0], attack_dm),
                                ('APGD20', attacks[1], attack_dm)]:
        started = time.monotonic()
        counts.update(examples=0, batches=0)
        before = state_digest(model.state_dict())
        metrics = trainer.test(model=module, datamodule=data, ckpt_path=None)
        after = state_digest(model.state_dict())
        expected = (640 if name == 'clean' else 256) if config['mode'] == 'control' else 10000
        if counts['examples'] != expected or counts['batches'] == 0 or before != after:
            raise ValueError('evaluation coverage or saved model/BN drift')
        if not metrics or any(not math.isfinite(float(v)) for row in metrics for v in row.values()):
            raise ValueError('missing/nonfinite metrics')
        record = {'kind': name, **counts, 'metrics': metrics, 'state_digest': after,
                  'checkpoint_sha256': sha256(final), 'seconds': time.monotonic()-started}
        write(root/('evaluation_'+name+'.json'), record)
        results.append(record)
    write(root/'evaluate.json', {'status': 'COMPLETED', 'config_sha256': sha256(path),
        'arm': arm, 'checkpoint_sha256': sha256(final), 'records': results,
        'seconds': time.monotonic()-began, 'full_test_set': config['mode'] == 'training',
        'empirical_only': True, 'formal_SAFE': False})


def audit(config, path, arm):
    began = time.monotonic()
    check_binding(config, path)
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, config['repo'])
    root = Path(config['output_root'])/arm
    tr = json.loads((root/'train.json').read_text())
    ev = json.loads((root/'evaluate.json').read_text())
    checkpoint = root/'final_epoch.ckpt'
    digest = sha256(checkpoint)
    if tr['checkpoint_sha256'] != digest or ev['checkpoint_sha256'] != digest:
        raise ValueError('final weight substitution')
    for record in (tr, ev):
        if record['config_sha256'] != sha256(path) or record['arm'] != arm or record['status'] != 'COMPLETED':
            raise ValueError('record binding/completeness')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['reproduction_binding'] != {'config_sha256': sha256(path), 'arm': arm}:
        raise ValueError('checkpoint provenance')
    if state_digest(state['state_dict']) != tr['state_digest'] or not all(
            torch.isfinite(v).all() for v in state['state_dict'].values()):
        raise ValueError('saved state discrepancy')
    if state['epoch'] != tr['epochs'] or state['global_step'] != tr['global_step']:
        raise ValueError('final cursor')
    if not state['reproduction_cuda_rng'] or not state['reproduction_rng'] or not state['lr_schedulers']:
        raise ValueError('missing recovery metadata; automatic resume remains disabled')
    momentum = [v['momentum_buffer'] for opt in state['optimizer_states']
                for v in opt['state'].values() if 'momentum_buffer' in v]
    if not momentum or not all(torch.isfinite(v).all() for v in momentum):
        raise ValueError('invalid optimizer state')
    if len(tr['epoch_checkpoints']) != tr['epochs']:
        raise ValueError('missing completed-epoch checkpoint')
    for name, h in tr['epoch_checkpoints'].items():
        if sha256(root/'checkpoints'/name) != h:
            raise ValueError('epoch checkpoint altered')
    if [r['kind'] for r in ev['records']] != ['clean', 'PGD20', 'APGD20']:
        raise ValueError('missing evaluation obligation')
    for record in ev['records']:
        if (json.loads((root/('evaluation_'+record['kind']+'.json')).read_text()) != record or
                record['state_digest'] != tr['state_digest'] or record['checkpoint_sha256'] != digest):
            raise ValueError('evaluation source or state drift')
    write(root/'audit.json', {'status': 'COMPLETED', 'config_sha256': sha256(path), 'arm': arm,
        'audit': 'SAVED_STATE_AND_EVALUATION_ROSTER_PASS', 'checkpoint_sha256': digest,
        'finite_momentum_buffers': len(momentum), 'seconds': time.monotonic()-began,
        'source_files': {str(root/name): sha256(root/name) for name in ['train.json','evaluate.json']},
        'claim': 'saved-state/coverage audit, not independent rerun of training or certification'})


def manage(config, path, arm):
    root = Path(config['output_root'])/arm
    for stage in STAGES:
        started = time.monotonic()
        write(root/(stage+'_started.json'), {'stage': stage, 'unix': time.time(),
            'config_sha256': sha256(path), 'arm': arm})
        with (root/(stage+'.stdout')).open('x') as out, (root/(stage+'.stderr')).open('x') as err:
            code = subprocess.run([config['python'], str(Path(__file__).resolve()), '--config', str(path.resolve()),
                '--arm', arm, '--stage', stage], stdout=out, stderr=err).returncode
        write(root/(stage+'_finished.json'), {'stage': stage, 'returncode': code,
            'seconds': time.monotonic()-started})
        if code:
            raise SystemExit(code)
    write(root/'inner_terminal.json', {'status': 'PIPELINE_COMPLETED', 'arm': arm,
        'config_sha256': sha256(path), 'stage_hashes': {s: sha256(root/(s+'.json')) for s in STAGES}})


def terminal(root, receipt, binding, arm, mode):
    """Fail closed on missing stages, changed identities, late or partial output."""
    root = Path(root)
    accepted, error = False, None
    try:
        if receipt['status'] == 'COMPLETED':
            inner = json.loads((root/'inner_terminal.json').read_text())
            if (inner['status'] != 'PIPELINE_COMPLETED' or inner['arm'] != arm or
                    inner['config_sha256'] != binding or set(inner['stage_hashes']) != set(STAGES)):
                raise ValueError('terminal identity/roster')
            for stage in STAGES:
                file = root/(stage+'.json')
                record = json.loads(file.read_text())
                finish = json.loads((root/(stage+'_finished.json')).read_text())
                if (sha256(file) != inner['stage_hashes'][stage] or record['status'] != 'COMPLETED' or
                        record['config_sha256'] != binding or record['arm'] != arm or
                        finish['returncode'] != 0):
                    raise ValueError('stage identity/completeness')
            accepted = True
    except (ValueError, KeyError, OSError, TypeError) as exc:
        error = repr(exc)
    return {'accepted': accepted, 'status': ('CONTROL_PIPELINE_AUDITED' if mode == 'control' else
        'FINAL_TRAINING_AND_EVALUATION_AUDITED') if accepted else
        ('ERROR' if receipt['status'] == 'COMPLETED' else receipt['status']),
        'config_sha256': binding, 'arm': arm, 'error': error,
        'execution_seconds': receipt['execution_including_preflight_seconds'],
        'with_postflight_seconds': receipt['total_with_postflight_seconds'],
        'audit_in_execution_budget': True, 'automatic_resume': False, 'formal_SAFE': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--arm', choices=['dense','convmoe'])
    p.add_argument('--stage', choices=['manager',*STAGES])
    args = p.parse_args()
    manifest = json.loads(args.config.read_text())
    if args.stage:
        if args.arm is None:
            raise ValueError('arm required for worker')
        if args.stage == 'manager':
            manage(manifest,args.config,args.arm)
        else:
            {'train':train,'evaluate':evaluate,'audit':audit}[args.stage](manifest,args.config,args.arm)
    else:
        launch_start = time.monotonic()
        check_binding(manifest, args.config)
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
            raise ValueError('commit/push freeze before real execution')
        if subprocess.check_output(['git','branch','--show-current'],text=True).strip() != 'feat/moe-route-verification':
            raise ValueError('wrong branch')
        root = Path(manifest['output_root'])
        root.mkdir(exist_ok=False)
        launch_guard_seconds = time.monotonic()-launch_start
        records=[]
        for arm in ['dense','convmoe']:
            receipt = supervise([manifest['python'],str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
                '--arm',arm,'--stage','manager'],str(Path(__file__).resolve().parents[1]),root/arm,
                manifest['seconds_per_arm'],'NATIVE_FULL_WORKFLOW',cpu_only=False)
            result = terminal(root/arm,receipt,sha256(args.config),arm,manifest['mode'])
            write(root/arm/'terminal.json',result)
            records.append(result)
            if not result['accepted']:
                break
        write(root/'summary.json',{'config_sha256':sha256(args.config),'records':records,
            'accepted':len(records)==2 and all(r['accepted'] for r in records),
            'launch_guard_seconds_outside_arm_budgets':launch_guard_seconds,
            'whole_launch_with_postflight_seconds':time.monotonic()-launch_start})
