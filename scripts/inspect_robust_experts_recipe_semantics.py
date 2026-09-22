"""No-training native controls for the frozen recipe's entropy and STE meaning.

Run in the existing author CPU environment, not by installing author packages
in ACT. No checkpoint, dataset, optimizer, inference or attack is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    began=time.monotonic()
    cfg=json.loads(args.config.read_text())
    repo=Path(cfg['repo'])
    paths=[repo/'src/models/nn/moe/gate/topk.py', repo/'src/models/nn/moe/routing.py',
           repo/'src/models/nn/moe/resnet_conv_moe.py',repo/'src/models/nn/moe/resnet_block_moe.py',
           repo/'src/models/lit_module.py',repo/'src/utils/attack.py',repo/'src/run.py',
           repo/'configs/experiment/cifar100-resnet18/default.yaml',
           repo/'configs/experiment/cifar100-resnet18/resnet_conv_moe.yaml']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    if any(cfg['files'].get(path)!=h for path,h in hashes.items()):
        raise ValueError('source differs from executed configuration')
    launcher=repo/'bash/cifar100_resnet18.sh'
    original=subprocess.check_output(['git','-C',str(repo),'show','HEAD:bash/cifar100_resnet18.sh'])
    if original!=launcher.read_bytes():
        raise ValueError('comparison-only author launcher changed from pinned commit')
    sys.path.insert(0,str(repo))
    import torch
    from src.models.nn.moe.gate.topk import _balancing_loss, TopKGate
    torch.set_num_threads(2)
    probabilities=torch.full((2,4),.25,dtype=torch.float64)
    native_entropy=float(_balancing_loss(probabilities,'entropy'))
    native_column=float(_balancing_loss(probabilities,'column_entropy'))
    # Uniform samples distinguish the objectives without a network experiment.
    expected=-float(torch.log(torch.tensor(4.,dtype=torch.float64)))
    if native_entropy!=0 or abs(native_column-expected)>1e-14:
        raise ValueError('formula control changed')
    gate=TopKGate(torch.nn.Identity(),k=2,use_straight_through_estimator=True,expert_capacity=None)
    if gate.use_straight_through_estimator:
        raise ValueError('native top2 STE behavior changed')
    result={'status':'SOURCE_SEMANTIC_CONTROLS_PASS',
            'scope':'read-only formula/constructor controls; NOT causal training or performance evidence',
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'author_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            'source_sha256':hashes, 'reviewer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'comparison_only_not_executed_launcher':{'path':str(launcher),
                'sha256':hashlib.sha256(original).hexdigest()},
            'uniform_two_by_four_control':{'source_entropy':native_entropy,'source_column_entropy':native_column,
                'paper_negative_entropy_of_batch_mean':expected},
            'top2_requested_ste':True,'top2_effective_ste':gate.use_straight_through_estimator,
            'training_runs':0,'model_queries':0,'seconds':time.monotonic()-began}
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
