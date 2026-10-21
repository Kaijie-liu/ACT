"""One frozen CPU-only twelve-model queue. Training time is separately charged."""
import time
ENTRY=time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha


def setup(threads):
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    import torch
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    return torch


def atomic_torch_save(path,value,torch):
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def control(folder):
    folder.mkdir()
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,max(.01,300-(time.monotonic()-ENTRY)))
    torch=setup(1)
    from act.pipeline.moe.multilayer_training import build,load_checkpoint
    from act.pipeline.moe.train import _balance_loss
    rows=[]
    for layers,experts in ((1,8),(2,8),(3,8),(2,16)):
        config=dict(input_dim=4,classes=3,hidden_output=5,router_hidden=[4],expert_hidden=[6,5],
                    layers=layers,experts=experts)
        model=build(config,0)
        x=torch.arange(16,dtype=torch.float32).reshape(4,4)/16
        y=torch.tensor([0,1,2,0])
        output,decisions=model.forward_with_routing(x)
        if not torch.equal(output,model(x)):raise ValueError('training/dynamic forward mismatch')
        loss=torch.nn.functional.cross_entropy(output,y)+.1*torch.stack(
            [_balance_loss(d,experts,'switch') for d in decisions]).mean()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        before=[p.detach().clone() for p in model.parameters()]
        loss.backward();optimizer.step()
        if not torch.isfinite(loss) or not any(not torch.equal(a,b) for a,b in zip(before,model.parameters())):
            raise ValueError('finite effective training step')
        path=folder/f'L{layers}E{experts}.pt'
        shuffle=torch.Generator().manual_seed(7)
        torch.randperm(4,generator=shuffle)
        atomic_torch_save(path,dict(format='act-sequential-top2-v1',architecture=config,seed=0,
            state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
            shuffle_rng=shuffle.get_state()),torch)
        restored,payload=load_checkpoint(path)
        if not torch.equal(model(x),restored(x)):raise ValueError('checkpoint restore changed output')
        # Resume one optimizer step from the same state; do not silently reinitialize moments.
        resumed=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.0001)
        resumed.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['rng'])
        restored_shuffle=torch.Generator();restored_shuffle.set_state(payload['shuffle_rng'])
        if not torch.equal(torch.randperm(4,generator=shuffle),torch.randperm(4,generator=restored_shuffle)):
            raise ValueError('serialized shuffle RNG differential')
        for network,opt in ((model,optimizer),(restored,resumed)):
            opt.zero_grad(set_to_none=True)
            value,decisions=network.forward_with_routing(x)
            objective=torch.nn.functional.cross_entropy(value,y)+.1*torch.stack(
                [_balance_loss(d,experts,'switch') for d in decisions]).mean()
            objective.backward();opt.step()
        if any(not torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters())):
            raise ValueError('optimizer resume differential')
        rows.append(dict(layers=layers,experts=experts,status='PASS',checkpoint_sha256=sha(path)))
    save(folder/'complete.json',dict(status='PASS',controls=rows,seconds=time.monotonic()-ENTRY,
        real_training=0,guarantee='SYNTHETIC_TRAIN_SAVE_RESTORE_ONLY'))
    signal.setitimer(signal.ITIMER_REAL,0.)


def train(folder,manifest):
    spec=load(manifest)
    if folder.exists():raise ValueError('new training directory')
    if ROOT==Path('/data1/Kane/MOE/ACT') or not (ROOT/'.frozen_commit').is_file():
        raise ValueError('immutable training code required')
    for p,digest in spec['source_files'].items():
        if sha(ROOT/p)!=digest:raise ValueError('frozen training source '+p)
    folder.mkdir()
    torch=setup(spec['threads'])
    from torch.utils.data import DataLoader,random_split
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from act.pipeline.moe.multilayer_training import build
    from act.pipeline.moe.train import _balance_loss
    control_result=load(spec['control_result'],spec['control_sha256'])
    if control_result['status']!='PASS':raise ValueError('training control not passed')
    for p,digest in spec['data_files'].items():
        if sha(p)!=digest:raise ValueError('fixed CIFAR data identity')
    dataset=CIFAR10(spec['dataset_root'],train=True,download=False,transform=ToTensor())
    train_data,val_data=random_split(dataset,[45000,5000],generator=torch.Generator().manual_seed(spec['split_seed']))
    split=dict(train=train_data.indices,validation=val_data.indices)
    save(folder/'split.json',split)
    split_hash=sha(folder/'split.json')
    save(folder/'start.json',dict(commit=(ROOT/'.frozen_commit').read_text().strip(),manifest_sha256=sha(manifest),
        split_sha256=split_hash,pid=os.getpid(),started_at=time.time(),training_seconds_separate=True))
    summary=[]
    @torch.no_grad()
    def evaluate(model,loader):
        model.eval();correct=total=0
        routes=[torch.zeros(len(layer.experts),dtype=torch.int64) for layer in model.layers]
        for x,y in loader:
            value,decisions=model.forward_with_routing(x)
            correct+=int((value.argmax(1)==y).sum());total+=len(y)
            for count,d in zip(routes,decisions):count.add_(torch.bincount(d.indices.flatten(),minlength=len(count)))
        return dict(accuracy=correct/total,correct=correct,total=total,route_counts=[r.tolist() for r in routes])
    for item in spec['models']:
        began=time.monotonic();target=folder/item['id'];target.mkdir()
        torch.manual_seed(item['seed'])
        model=build(item['architecture'],item['seed'])
        loader=DataLoader(train_data,batch_size=256,shuffle=True,num_workers=0,
            generator=torch.Generator().manual_seed(item['seed']))
        validation=DataLoader(val_data,batch_size=256,shuffle=False,num_workers=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best_accuracy=-1.;best_state=None;best_epoch=None;best_metrics=None
        for epoch in range(1,spec['epochs']+1):
            epoch_start=time.monotonic();model.train();running=correct=total=0
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True)
                value,decisions=model.forward_with_routing(x)
                balance=torch.stack([_balance_loss(d,item['architecture']['experts'],'switch') for d in decisions]).mean()
                loss=torch.nn.functional.cross_entropy(value,y)+.1*balance
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();optimizer.step()
                running+=float(loss.detach())*len(y);correct+=int((value.detach().argmax(1)==y).sum());total+=len(y)
            metrics=evaluate(model,validation)
            if metrics['accuracy']>best_accuracy:
                best_accuracy=metrics['accuracy'];best_epoch=epoch;best_metrics=metrics
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            row=dict(epoch=epoch,loss=running/total,train_accuracy=correct/total,validation=metrics,
                     seconds=time.monotonic()-epoch_start,model_seconds=time.monotonic()-began)
            save(target/f'epoch_{epoch:03d}.json',row)
            atomic_torch_save(target/'resume.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                best_state=best_state,best_epoch=best_epoch,best_metrics=best_metrics,epoch=epoch,
                rng=torch.get_rng_state(),shuffle_rng=loader.generator.get_state(),item=item,
                manifest_sha256=sha(manifest),split_sha256=split_hash),torch)
            print(json.dumps(dict(model=item['id'],**row)),flush=True)
        model.load_state_dict(best_state)
        test=CIFAR10(spec['dataset_root'],train=False,download=False,transform=ToTensor())
        test_metrics=evaluate(model,DataLoader(test,batch_size=256,shuffle=False,num_workers=0))
        path=target/'selected.pt'
        atomic_torch_save(path,dict(format='act-sequential-top2-v1',architecture=item['architecture'],seed=item['seed'],
            state_dict=best_state,best_epoch=best_epoch,validation=best_metrics,test=test_metrics,
            split_sha256=split_hash,manifest_sha256=sha(manifest),code_commit=(ROOT/'.frozen_commit').read_text().strip()),torch)
        result=dict(id=item['id'],best_epoch=best_epoch,validation=best_metrics,test=test_metrics,
                    checkpoint=str(path),checkpoint_sha256=sha(path),seconds=time.monotonic()-began)
        save(target/'complete.json',result);summary.append(result)
    save(folder/'complete.json',dict(status='COMPLETE',models=summary,seconds=time.monotonic()-ENTRY,
                                   verification_calls=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path)
    parser.add_argument('--run',type=Path);parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    try:
        if args.control:control(args.control)
        else:train(args.run,args.manifest)
    except BaseException as exc:
        folder=args.control or args.run
        if folder.exists():save(folder/'error.json',dict(kind=type(exc).__name__,message=str(exc),
                              traceback=traceback.format_exc(),seconds=time.monotonic()-ENTRY))
        raise
