"""Bounded native Lightning/PGD pipeline, isolated paths and local-only logging.

No rewritten optimizer, loss, gate, normalization or attack. CPU compatibility
stack and data_dir forwarding are explicit. A one-batch control is NOT training
the paper model or establishing an empirical robustness rate.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from recent_moe_deployment import sha256, supervise
from recent_moe_env_inventory import inventory


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def terminal(receipt, result):
    accepted = bool(receipt['status'] == 'COMPLETED' and result and
                    result.get('status') == 'NATIVE_WORKFLOW_CONTROL_PASS')
    return {'status': result['status'] if accepted else
            ('ERROR' if receipt['status'] == 'COMPLETED' else receipt['status']),
            'accepted': accepted, 'execution_seconds': receipt['execution_including_preflight_seconds'],
            'with_postflight_seconds': receipt['total_with_postflight_seconds'], 'formal_SAFE': False}


def compose(repo, root, data):
    import hydra
    from omegaconf import OmegaConf, open_dict
    sys.path.insert(0, str(repo))
    from src import run as native
    with hydra.initialize_config_dir(config_dir=str(repo / 'configs'), version_base='1.1'):
        cfg = hydra.compose(config_name='config', overrides=[
            'experiment=cifar100-resnet18/pgd_adv_train_resnet_conv_moe',
            'logger=csv', 'callbacks=default'])
    with open_dict(cfg):
        cfg.work_dir = str(root)
        cfg.data_dir = str(data)
        cfg.use_clearml = False
        cfg.wandb = {}
        cfg.print_config = False
        cfg.logger = {'csv': {'_target_': 'pytorch_lightning.loggers.CSVLogger',
                              'save_dir': str(root), 'name': 'local_metrics', 'version': 'control'}}
        cfg.callbacks = {'checkpoint': {'_target_': 'pytorch_lightning.callbacks.ModelCheckpoint',
            'dirpath': str(root / 'checkpoints'), 'save_last': True, 'save_top_k': 0}}
        cfg.datamodule.data_dir = str(data)
        cfg.datamodule.batch_size = 2
        cfg.datamodule.num_workers = 0
        cfg.datamodule.pin_memory = False
        cfg.trainer.gpus = 0
        cfg.trainer.min_epochs = 1
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 1
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.limit_test_batches = 1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.enable_model_summary = False
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.default_root_dir = str(root)
        cfg.execute_train = True
        cfg.execute_test = True
        cfg.execute_attack = True
        cfg.attack_batch_size = 2
    if (cfg.model.model.k != 1 or cfg.model.model.num_experts != 4
            or cfg.model.attack.steps != 7 or cfg.model.attack.eps != .03137
            or cfg.model.optimizer.lr != .1 or cfg.seed != 12345):
        raise ValueError('unexpected author recipe')
    return native, cfg, OmegaConf.to_container(cfg, resolve=False)


def worker(manifest, config):
    start = time.monotonic()
    for file, h in manifest['files'].items():
        if sha256(file) != h:
            raise ValueError('frozen source/data/config changed: ' + file)
    if inventory([manifest['python']]) != json.loads(Path(manifest['environment']).read_text()):
        raise ValueError('workflow environment changed')
    import socket
    def forbidden(*args, **kwargs):
        raise RuntimeError('network connection prohibited in this frozen workflow control')
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['CLEARML_OFFLINE_MODE'] = '1'
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    repo, root, data = map(Path, [manifest['repo'], manifest['output_root'], manifest['data_root']])
    native, cfg, composed = compose(repo, root, data)
    if composed != manifest['composed_config']:
        raise ValueError('configuration composition differs')
    write(root / 'prepared.json', {'config_sha256': sha256(config), 'setup_seconds': time.monotonic()-start,
                                  'composed_config': composed})
    original_setup = native._setup
    captures = {'test_calls': []}
    def located_setup(configured):
        result = original_setup(configured)
        attacks, checkpoint, dm, model, trainer = result
        if any(attack.model is not model.model for attack in attacks):
            raise ValueError('native attack wrappers do not share the trained model')
        # Author subclass consumes data_dir without forwarding it. Restrict the
        # location after construction; do not alter transforms/splits/loaders.
        dm.data_dir = str(data)
        captures.update({'model': model, 'trainer': trainer,
            'before': {n: p.detach().clone() for n, p in model.named_parameters()}})
        def invalid_grad(_mod, _input, _output):
            if isinstance(_output, torch.Tensor) and not torch.isfinite(_output).all():
                raise ValueError('nonfinite complete-model output')
        model.model.register_forward_hook(invalid_grad)
        original_test = trainer.test
        def recorded_test(*args, **kwargs):
            began = time.monotonic()
            values = original_test(*args, **kwargs)
            record = {'call': len(captures['test_calls']), 'seconds': time.monotonic()-began,
                      'metrics': values}
            captures['test_calls'].append(record)
            write(root / f'test_call{record["call"]}.json', record)
            return values
        trainer.test = recorded_test
        # Observe the existing Lightning hook; do not modify or clip gradients.
        original_before_step = model.on_before_optimizer_step
        def finite_step(optimizer, optimizer_idx):
            original_before_step(optimizer, optimizer_idx)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError('invalid gradients before native optimizer update')
        model.on_before_optimizer_step = finite_step
        return result
    native._setup = located_setup
    # Native run also constructs another attack datamodule. Forward only the
    # swallowed path for every author CIFAR100DataModule instance.
    from src.datamodules.cifar100_datamodule import CIFAR100DataModule
    original_init = CIFAR100DataModule.__init__
    def located_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.data_dir = str(data)
    CIFAR100DataModule.__init__ = located_init
    train_start = time.monotonic()
    native.run(cfg)
    model, trainer = captures['model'], captures['trainer']
    changed = sum(not torch.equal(captures['before'][n], p) for n, p in model.named_parameters())
    if (trainer.global_step != 1 or not changed or len(captures['test_calls']) != 3
            or not all(torch.isfinite(p).all() for p in model.parameters())):
        raise ValueError('one finite native training update not completed')
    # Tensor-only inference reload, NOT a claim of full Lightning resume.
    saved = root / 'trained_state.pt'
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, saved)
    reloaded = torch.load(saved, weights_only=True, map_location='cpu')
    if state.keys() != reloaded.keys() or not all(torch.equal(v, reloaded[k]) for k, v in state.items()):
        raise ValueError('saved prediction-state mismatch')
    write(root / 'result.json', {'status': 'NATIVE_WORKFLOW_CONTROL_PASS', 'config_sha256': sha256(config),
        'global_steps': trainer.global_step, 'changed_parameter_tensors': changed,
        'test_calls': captures['test_calls'], 'order': ['clean', 'PGD20', 'APGD20'],
        'prediction_state_reload_exact': True, 'state_sha256': sha256(saved),
        'setup_seconds': train_start-start, 'native_workflow_save_seconds': time.monotonic()-train_start,
        'worker_total_seconds': time.monotonic()-start, 'evidence_grade': 'TRAIN_ATTACK_DEPLOYMENT_CONTROL',
        'formal_SAFE': False,
        'scope': 'one train/validation/test batch and native PGD20/APGD20 batches, NOT paper training/accuracy or exact resume'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.worker:
        worker(cfg, a.config)
    else:
        root = Path(cfg['output_root'])
        receipt = supervise([cfg['python'], str(Path(__file__).resolve()), '--config', str(a.config.resolve()), '--worker'],
            str(Path(__file__).resolve().parents[1]), root, cfg['total_seconds'],
            'ROBUST_EXPERTS_NATIVE_WORKFLOW', cfg['original_repo'])
        path = root / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else None
        write(root / 'terminal.json', terminal(receipt, result))
