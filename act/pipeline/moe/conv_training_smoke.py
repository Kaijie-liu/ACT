"""Two real augmented batches on CUDA; discarded control, never production state."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch
from torch.utils.data import Subset
from act.back_end.moe.conv_factory import ConvOutputMoEConfig,build_conv_output_moe
from act.back_end.moe.factory import load_output_moe_checkpoint
from act.pipeline.moe import conv_training as worker
from act.util.path_config import get_torchvision_data_root


def run(root,device):
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    cfg=json.loads(worker.CONFIG.read_text())
    result=dict(status='RUNNING',device=device,config_sha256=worker.sha(worker.CONFIG),
        worker_sha256=worker.sha(Path(worker.__file__)),scope='discarded two-batch training control, not model selection')
    try:
        worker.setup(17,2,device)
        tr,va,_=worker.datasets(Path(get_torchvision_data_root())/'CIFAR10/raw',cfg)
        ids,_=worker.split_indices(50000,.1,17)
        # Separate transform objects: no random validation transform inheritance.
        if not torch.equal(va[ids[0]][0],va[ids[0]][0]):raise ValueError('validation augmentation leak')
        result['augmentation']=str(tr.transform)
        if torch.equal(tr[ids[0]][0],tr[ids[0]][0]):raise ValueError('augmentation control unchanged')
        model=build_conv_output_moe(ConvOutputMoEConfig(**cfg['factory'])).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100)
        metrics=worker.train_epoch(model,worker.loader(Subset(tr,ids[:256]),cfg,device,18,True),optimizer,device,.01)
        scheduler.step()
        if metrics['router_max_update']<=0 or metrics['router_max_gradient_l1']<=0:raise ValueError('router not trained')
        model.eval();point=va[ids[0]][0].unsqueeze(0).to(device)
        with torch.no_grad():reference=model(point).clone()
        path=root/'control.pt'
        worker.save_checkpoint(path,dict(format='act-output-conv-moe-v1',factory_config=cfg['factory'],
            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()))
        restored,payload=load_output_moe_checkpoint(path,map_location='cpu');restored.to(device).eval()
        with torch.no_grad():error=float((restored(point)-reference).abs().max())
        if error!=0:raise ValueError('checkpoint replay not bitwise equal')
        opt2=torch.optim.AdamW(restored.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,100)
        opt2.load_state_dict(payload['optimizer']);sched2.load_state_dict(payload['scheduler'])
        # Verify one identical post-reload optimizer update, not a claim that
        # arbitrary interrupted mid-epoch training can be automatically resumed.
        batch=[(point.cpu(),torch.tensor([va[ids[0]][1]]))]
        worker.train_epoch(model,batch,optimizer,device,.01)
        worker.train_epoch(restored,batch,opt2,device,.01)
        if any(not torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values())):
            raise ValueError('optimizer checkpoint continuation mismatch')
        result.update(status='PASS',metrics=metrics,checkpoint_replay_max_error=error,
            optimizer_continuation_equal=True,next_learning_rate=sched2.get_last_lr()[0],
            peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
            torch=torch.__version__)
    except Exception as exc:result.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-start;worker.atomic_json(root/'summary.json',result)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=p.parse_args()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8')
    if run(args.output.resolve(),args.device)['status']!='PASS':raise SystemExit(1)
