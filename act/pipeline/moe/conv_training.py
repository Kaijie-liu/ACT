"""Frozen convolutional R1 training worker and read-only landing audit.

Run through conv_training_supervisor. No automatic retry, checkpoint replacement,
early stopping, verification-dependent selection or test-dependent selection.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from act.back_end.moe.conv_factory import ConvOutputMoEConfig, build_conv_output_moe
from act.back_end.moe.factory import load_output_moe_checkpoint
from act.pipeline.moe.train import _balance_loss, evaluate

CONFIG = Path(__file__).parent / 'configs/conv_family_training_r1.json'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def save_checkpoint(path, payload):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temp=path.with_suffix('.pt.tmp')
    with temp.open('wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def split_indices(total, fraction, seed):
    order=torch.randperm(total,generator=torch.Generator().manual_seed(seed)).tolist()
    count=round(total*fraction)
    if not 0<count<total:raise ValueError('invalid split')
    return order[:-count],order[-count:]


def selected_epoch(rows):
    if not rows:raise ValueError('empty history')
    # Integer correct count, fixed validation denominator, earliest exact tie.
    return min(rows,key=lambda row:(-row['validation_correct'],row['epoch']))['epoch']


def seed_worker(_):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)


def setup(seed,threads,device):
    torch.set_default_dtype(torch.float32);torch.set_num_threads(threads)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # No AMP or silent precision fallback. Deterministic algorithms fail closed.
    torch.use_deterministic_algorithms(True)
    if device=='cuda':torch.cuda.set_device(0)


def datasets(data_root, cfg):
    from torchvision.datasets import CIFAR10
    from torchvision import transforms as T
    aug=cfg['augmentation']
    train=CIFAR10(str(data_root),train=True,download=False,transform=T.Compose([
        T.RandomCrop(32,padding=aug['random_crop_padding']),
        T.RandomHorizontalFlip(p=aug['horizontal_flip_probability']),T.ToTensor()]))
    validation=CIFAR10(str(data_root),train=True,download=False,transform=T.ToTensor())
    test=CIFAR10(str(data_root),train=False,download=False,transform=T.ToTensor())
    if len(train)!=50000 or len(test)!=10000:raise ValueError('CIFAR10 length mismatch')
    return train,validation,test


def loader(dataset,cfg,device,seed,shuffle=False):
    return DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=shuffle,
        num_workers=cfg['workers'],worker_init_fn=seed_worker,persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),pin_memory=device=='cuda')


def train_epoch(model,batches,optimizer,device,coefficient,heartbeat=None):
    model.train();total=correct=0;loss_sum=ce_sum=balance_sum=0.;routes=torch.zeros(model.spec.num_experts,dtype=torch.long)
    initial=[p.detach().clone() for p in model.router.parameters()]
    max_router_grad=0.
    for batch,(x,y) in enumerate(batches,1):
        x=x.to(device);y=y.to(device)
        optimizer.zero_grad(set_to_none=True)
        output,decision=model.forward_with_routing(x)
        ce=torch.nn.functional.cross_entropy(output,y)
        balance=_balance_loss(decision,model.spec.num_experts,'switch')
        loss=ce+coefficient*balance
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('nonfinite gradient')
        max_router_grad=max(max_router_grad,sum(float(p.grad.detach().abs().sum()) for p in model.router.parameters() if p.grad is not None))
        optimizer.step()
        n=y.numel();total+=n;correct+=int((output.argmax(-1)==y).sum())
        loss_sum+=float(loss.detach())*n;ce_sum+=float(ce.detach())*n;balance_sum+=float(balance.detach())*n
        routes+=torch.bincount(decision.indices.detach().cpu().reshape(-1),minlength=model.spec.num_experts)
        if heartbeat and (batch==1 or batch%25==0):heartbeat(batch,total)
    if not total:raise ValueError('empty epoch')
    delta=max(float((p.detach()-old).abs().max()) for p,old in zip(model.router.parameters(),initial))
    return dict(samples=total,correct=correct,accuracy=correct/total,loss=loss_sum/total,
        classification_loss=ce_sum/total,balance_loss=balance_sum/total,route_counts=routes.tolist(),
        router_max_gradient_l1=max_router_grad,router_max_update=delta)


def provenance(root):
    launch=json.loads((root/'launch.json').read_text())
    if sha(CONFIG)!=launch['config_sha256'] or sha(root/'config.json')!=launch['config_sha256']:
        raise ValueError('training recipe drift')
    for p,digest in launch['dataset_hashes'].items():
        if sha(p)!=digest:raise ValueError('dataset identity drift')
    return launch,json.loads((root/'config.json').read_text())


def run(root,device):
    launch,cfg=provenance(root)
    setup(cfg['factory']['seed'],cfg['cpu_threads'],device)
    train_data,val_data,test_data=datasets(launch['dataset_root'],cfg)
    train_ids,val_ids=split_indices(len(train_data),cfg['validation_fraction'],cfg['split_seed'])
    atomic_json(root/'split.json',dict(train=train_ids,validation=val_ids,seed=cfg['split_seed'],test_used_for_selection=False))
    net=build_conv_output_moe(ConvOutputMoEConfig(**cfg['factory'])).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=0.)
    history=[];start=time.monotonic()
    def heartbeat(phase,epoch,**extra):
        atomic_json(root/'heartbeat.json',dict(status='RUNNING',phase=phase,epoch=epoch,updated=time.time(),**extra))
    for epoch in range(1,cfg['epochs']+1):
        tick=time.monotonic();heartbeat('train',epoch)
        # Epoch-local generators make worker/shuffle seeds explicit; states are
        # also preserved in each immutable checkpoint. No automatic resume.
        train_loader=loader(Subset(train_data,train_ids),cfg,device,cfg['split_seed']+epoch,True)
        lr=opt.param_groups[0]['lr']
        metrics=train_epoch(net,train_loader,opt,device,cfg['balance_coefficient'],
            lambda batch,samples:heartbeat('train',epoch,batch=batch,samples=samples))
        heartbeat('validation',epoch)
        validation=evaluate(net,loader(Subset(val_data,val_ids),cfg,device,10000+epoch),torch.device(device))
        scheduler.step()
        row=dict(epoch=epoch,learning_rate=lr,next_learning_rate=opt.param_groups[0]['lr'],train=metrics,
            validation=validation,validation_correct=round(validation['accuracy']*len(val_ids)),
            seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-start,
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None)
        checkpoint=root/'checkpoints'/f'epoch_{epoch:03d}.pt'
        save_checkpoint(checkpoint,dict(format='act-output-conv-moe-v1',factory_config=cfg['factory'],
            state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,metrics=row,
            config_sha256=launch['config_sha256'],split_sha256=sha(root/'split.json'),
            rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device=='cuda' else [],
                 'python':random.getstate(),'numpy':np.random.get_state()},
            train_loader_generator=train_loader.generator.get_state()))
        row.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        atomic_json(root/'epochs'/f'{epoch:03d}.json',row);history.append(row)
        atomic_json(root/'selection.json',dict(best_epoch=selected_epoch(history),completed_epoch=epoch,rule=cfg['checkpoint_rule']))
        heartbeat('epoch_committed',epoch)
        print(json.dumps(row),flush=True)
    best=selected_epoch(history);chosen=history[best-1]
    selected,_=load_output_moe_checkpoint(chosen['checkpoint'],map_location='cpu')
    selected.to(device).eval();heartbeat('selected_test',cfg['epochs'])
    test=evaluate(selected,loader(test_data,cfg,device,20000),torch.device(device))
    summary=dict(status='TRAINING_COMPLETE_PENDING_AUDIT',epochs=cfg['epochs'],best_epoch=best,
        checkpoint=chosen['checkpoint'],checkpoint_sha256=chosen['checkpoint_sha256'],validation=chosen['validation'],
        test=test,seconds=time.monotonic()-start,config_sha256=launch['config_sha256'])
    atomic_json(root/'training_summary.json',summary);heartbeat('training_complete',cfg['epochs'])


def audit(root,device):
    launch,cfg=provenance(root);setup(cfg['factory']['seed'],cfg['cpu_threads'],device)
    rows=[json.loads((root/'epochs'/f'{e:03d}.json').read_text()) for e in range(1,cfg['epochs']+1)]
    split=json.loads((root/'split.json').read_text());a,b=split_indices(50000,cfg['validation_fraction'],cfg['split_seed'])
    if split['train']!=a or split['validation']!=b or set(a)&set(b) or len(set(a+b))!=50000:
        raise ValueError('split reconstruction failed')
    if len(list((root/'epochs').glob('*.json')))!=cfg['epochs']:raise ValueError('extra epoch rows')
    for e,row in enumerate(rows,1):
        checkpoint=root/'checkpoints'/f'epoch_{e:03d}.pt'
        expected_lr=cfg['learning_rate']*(1+math.cos(math.pi*(e-1)/cfg['epochs']))/2
        if (row['epoch']!=e or Path(row['checkpoint'])!=checkpoint or sha(checkpoint)!=row['checkpoint_sha256']
                or not math.isclose(row['learning_rate'],expected_lr,rel_tol=1e-12,abs_tol=1e-15)):
            raise ValueError('epoch identity or LR mismatch')
        if row['train']['samples']!=45000 or row['validation']['samples']!=5000 or sum(row['train']['route_counts'])!=90000:
            raise ValueError('epoch denominator mismatch')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['epoch']!=e or state['config_sha256']!=launch['config_sha256'] or state['split_sha256']!=sha(root/'split.json'):
            raise ValueError('checkpoint provenance mismatch')
        if state['metrics']!={k:v for k,v in row.items() if k not in ('checkpoint','checkpoint_sha256')}:
            raise ValueError('checkpoint metrics mismatch')
        if any(not torch.isfinite(v).all() for v in state['state_dict'].values()):raise ValueError('nonfinite checkpoint')
        if row['validation_correct']!=round(row['validation']['accuracy']*5000):raise ValueError('invalid accuracy count')
    best=selected_epoch(rows);summary=json.loads((root/'training_summary.json').read_text())
    if summary['best_epoch']!=best or summary['checkpoint_sha256']!=rows[best-1]['checkpoint_sha256']:
        raise ValueError('selection not earliest validation maximum')
    model,_=load_output_moe_checkpoint(rows[best-1]['checkpoint'],map_location='cpu');model.to(device).eval()
    _,validation,test=datasets(launch['dataset_root'],cfg)
    vm=evaluate(model,loader(Subset(validation,b),cfg,device,10000+best),torch.device(device))
    tm=evaluate(model,loader(test,cfg,device,20000),torch.device(device))
    if vm!=rows[best-1]['validation'] or tm!=summary['test']:raise ValueError('independent evaluation replay mismatch')
    result=dict(status='PASS',issues=[],epochs=len(rows),selected_epoch=best,
        checkpoint_sha256=summary['checkpoint_sha256'],validation=vm,test=tm,
        scope='training identity, full epoch denominators, recipe, selection and independent concrete metric replay; no verification claim')
    atomic_json(root/'audit.json',result)
    atomic_json(root/'CONV_LANDED_summary.json',{**summary,'status':'LANDED_AUDITED','audit':result,
        'audit_sha256':sha(root/'audit.json'),'launch_sha256':sha(root/'launch.json'),
        'split_sha256':sha(root/'split.json')})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--audit',action='store_true')
    args=p.parse_args()
    if args.audit:audit(args.root.resolve(),args.device)
    else:run(args.root.resolve(),args.device)
