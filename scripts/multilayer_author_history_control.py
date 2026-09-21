"""Frozen original-author static-history compatibility; NOT a box certificate.

Launcher uses act-py312; author-only forward/compiler check uses its already
installed environment. ACT HZ tests/solves remain in act-py312. No installation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(config, output):
    import torch
    sys.path.insert(0,config['author_repository'])
    from act.back_end.moe.multilayer import HistoryPlan
    from src.models.nn.moe.resnet_conv_moe import ResNetConvMoE
    from src.models.nn.moe.routing import GlobalAvgLinearRoutingNetwork
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    repo=Path(config['author_repository'])
    sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.rglob('*.py') if '__pycache__' not in p.parts}
    model=ResNetConvMoE(layers=18,num_channels=3,small_inputs=True,num_classes=100,
        num_experts=4,k=1,routing_layer_type=GlobalAvgLinearRoutingNetwork,expert_capacity=None,
        balancing_loss_type='entropy',balancing_loss=.5,moe_layer_prefix='layer4').double().eval()
    records=[]
    for value in config['probe_constants']:
        x=torch.full((1,3,32,32),value,dtype=torch.float64)
        actual=[]
        handles=[]
        for m in model.modules():
            if type(m).__name__=='TopKGate':
                handles.append(m.register_forward_hook(lambda module,args,out:actual.append(tuple(torch.nonzero(out[1][0],as_tuple=True)[0].tolist()))))
        with torch.no_grad(): expected=model(x)
        for h in handles: h.remove()
        plan=HistoryPlan(model,x,source_hashes=sources)
        if len(plan.sites)!=config['expected_sites'] or plan.total_histories!=config['expected_histories']:
            raise ValueError('unexpected full-size route-history catalog')
        compiled,score_rows=plan.compile(actual)
        with torch.no_grad(): found=compiled(x)
        error=(found[:,:100]-expected).abs().max().item()
        if not torch.allclose(found[:,:100],expected,rtol=0.,atol=config['concrete_tolerance']):
            raise ValueError('static history/native output mismatch')
        legal=all(all(found[0,rr[i]]>=found[0,rr[j]] for i in choice for j in range(4) if j not in choice)
                  for choice,rr in zip(actual,score_rows))
        if not legal: raise ValueError('native history fails mathematical guard at probe')
        records.append({'probe_constant':value,'history':[list(v) for v in actual],
            'maximum_absolute_error':error,'all_guards_hold':True,'compiled_nodes':len(list(compiled.graph.nodes)),
            'compiled_graph_sha256':hashlib.sha256(compiled.code.encode()).hexdigest()})
    # Source binding is mandatory, not a best-effort warning.
    bad=dict(sources)
    bad[next(p for p in bad if p.endswith('/gate/topk.py'))]='0'*64
    try: HistoryPlan(model,x,source_hashes=bad)
    except ValueError: rejected=True
    else: raise ValueError('wrong source accepted')
    result={'status':'COMPATIBILITY_PASS','records':records,'sites':5,'histories':1024,
        'parameters':sum(p.numel() for p in model.parameters()),'source_hashes':sources,
        'wrong_source_rejected':rejected,'box_certificate':False,'trained_checkpoint':False,
        'scope':'full-size original E4/k1/layer4 architecture; two finite probes, NOT all 1024 histories solved',
        'torch':torch.__version__,'python':sys.version}
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    if args.worker:
        worker(config,args.output_dir/'candidate.json')
        return
    began=time.monotonic()
    output=args.output_dir.resolve()
    if not output.is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('output scope')
    output.mkdir(parents=True,exist_ok=False)
    paths=[args.config.resolve(),Path(__file__).resolve(),ROOT/'act/back_end/moe/multilayer.py']
    identity={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    status='ERROR'
    with (output/'stdout.log').open('w') as out,(output/'stderr.log').open('w') as err:
        try:
            run=subprocess.run([config['author_python'],str(Path(__file__).resolve()),'--worker',
                '--config',str(args.config.resolve()),'--output-dir',str(output)],env=env,stdout=out,stderr=err,
                timeout=max(.001,config['total_seconds']-(time.monotonic()-began)))
            if run.returncode==0:
                result=json.loads((output/'candidate.json').read_text())
                if result['status']=='COMPATIBILITY_PASS' and not result['box_certificate'] and len(result['records'])==2:
                    status='COMPATIBILITY_PASS'
        except subprocess.TimeoutExpired: status='TIMEOUT'
    if time.monotonic()-began>=config['total_seconds']: status='TIMEOUT'
    summary={'status':status,'seconds':time.monotonic()-began,'budget_seconds':config['total_seconds'],
        'identity':identity,'candidate_sha256':hashlib.sha256((output/'candidate.json').read_bytes()).hexdigest()
            if (output/'candidate.json').exists() else None,'box_certificate':False}
    (output/'terminal.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    print(json.dumps(summary,sort_keys=True))
    if status!='COMPATIBILITY_PASS': raise SystemExit(1)


if __name__=='__main__': main()
