"""Frozen full-batch native PGD7 update/save on BOTH paper-recipe architectures.

No full training, evaluation or model selection; no smaller-batch fallback.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from recent_moe_deployment import sha256, supervise
from recent_moe_env_inventory import inventory
from robust_experts_workflow_control import write


def configured(recipe, root, data):
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.create(recipe)
    with open_dict(cfg):
        cfg.work_dir, cfg.data_dir = str(root), str(data)
        cfg.datamodule.data_dir = str(data)
        cfg.execute_train, cfg.execute_test, cfg.execute_attack = True, False, False
        cfg.optimized_metric = None
        cfg.print_config = False
        cfg.logger = {'csv': {'_target_':'pytorch_lightning.loggers.CSVLogger',
            'save_dir':str(root), 'name':'local', 'version':'step'}}
        cfg.callbacks = {'checkpoint': {'_target_':'pytorch_lightning.callbacks.ModelCheckpoint',
            'dirpath':str(root/'checkpoints'), 'save_last':True, 'save_top_k':0}}
        cfg.trainer.min_epochs = 0
        cfg.trainer.max_epochs = 200
        cfg.trainer.max_steps = 1
        cfg.trainer.limit_train_batches = 1
        cfg.trainer.limit_val_batches = 0
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.enable_model_summary = False
        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.default_root_dir = str(root)
    if (cfg.datamodule.batch_size != 640 or cfg.model.optimizer.lr != .01 or
            cfg.model.attack.steps != 7 or cfg.trainer.gpus != 1 or cfg.seed != 12345):
        raise ValueError('scientific recipe changed')
    return cfg


def worker(manifest, path, arm):
    start = time.monotonic()
    for p,h in manifest['files'].items():
        if sha256(p) != h:
            raise ValueError('frozen input changed: '+p)
    if inventory([manifest['python']]) != manifest['environment']:
        raise ValueError('GPU dependency drift')
    import socket
    def forbidden(*args, **kwargs):
        raise RuntimeError('network prohibited in GPU control')
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    os.environ.update(WANDB_MODE='disabled', CLEARML_OFFLINE_MODE='1')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise ValueError('GPU unavailable; no CPU fallback')
    free,total = torch.cuda.mem_get_info(0)
    if free < 48*1024**3:
        raise ValueError('GPU_RESOURCE_WAIT_REQUIRED; no eviction or automatic retry')
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    sys.path.insert(0, manifest['repo'])
    from src import run as native
    import pytorch_lightning as pl
    root = Path(manifest['output_root'])/arm
    data = Path(manifest['data_root'])
    cfg = configured(manifest['recipes'][arm], root, data)
    from omegaconf import OmegaConf
    write(root/'prepared.json', {'config_sha256':sha256(path), 'arm':arm,
        'resolved_config':OmegaConf.to_container(cfg,resolve=True),
        'gpu':torch.cuda.get_device_name(0), 'initial_free_bytes':free,
        'memory_fraction_cap':.4, 'total_gpu_bytes':total})
    _,_,dm,model,trainer = native._setup(cfg)
    dm.data_dir = str(data)  # author's subclass fails to forward location
    before = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    observed = []
    class Observer(pl.Callback):
        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            if len(batch[0]) != 640 or batch[0].device.type != 'cuda':
                raise ValueError('not the frozen full GPU batch')
            observed.append({'batch':batch_idx,'size':len(batch[0]),'device':str(batch[0].device)})
        def on_before_optimizer_step(self, trainer, model, optimizer, opt_idx):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('nonfinite/no native gradients')
    trainer.callbacks.append(Observer())
    trainer.fit(model,datamodule=dm)
    torch.cuda.synchronize()
    if trainer.global_step != 1 or len(observed) != 1:
        raise ValueError('exactly one native update required')
    changed = sum(not torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters())
    if not changed or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('unchanged/nonfinite weights')
    saved = root/'full_step.ckpt'
    trainer.save_checkpoint(saved)
    payload = torch.load(saved,map_location='cpu',weights_only=False)
    expected = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    if set(payload['state_dict']) != set(expected) or not all(
            torch.equal(v,payload['state_dict'][k]) for k,v in expected.items()):
        raise ValueError('saved model state mismatch')
    if not payload['optimizer_states'] or not payload['lr_schedulers'] or payload['global_step'] != 1:
        raise ValueError('missing recovery components')
    write(root/'result.json', {'status':'NATIVE_GPU_FULL_BATCH_STEP_PASS','arm':arm,
        'config_sha256':sha256(path),'batch_observations':observed,'steps':1,
        'changed_parameter_tensors':changed,'state_tensors':len(expected),
        'checkpoint_sha256':sha256(saved),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'seconds':time.monotonic()-start,'long_training':False,'exact_gpu_resume':False,
        'scope':'native full-batch PGD7/SGD and saved state; no accuracy or certification'})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--worker',choices=['dense','convmoe'])
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    if a.worker:
        worker(cfg,a.config,a.worker)
    else:
        root=Path(cfg['output_root']);root.mkdir(exist_ok=False)
        records=[]
        for arm in ['dense','convmoe']:
            rec=supervise([cfg['python'],str(Path(__file__).resolve()),'--config',str(a.config.resolve()),
                '--worker',arm],str(Path(__file__).resolve().parents[1]),root/arm,600,
                'GPU_DEPLOYMENT_ONLY',cpu_only=False)
            result=root/arm/'result.json'
            records.append({'arm':arm,'receipt':rec,
                'result':json.loads(result.read_text()) if result.exists() else None})
            if rec['status'] != 'COMPLETED':
                break
        write(root/'summary.json',{'config_sha256':sha256(a.config),'records':records,
            'accepted':len(records)==2 and all(r['receipt']['status']=='COMPLETED' and r['result']
                and r['result']['status']=='NATIVE_GPU_FULL_BATCH_STEP_PASS' for r in records)})
