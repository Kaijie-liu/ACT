"""Fixed analytic whole-box control; all four two-layer histories are required."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_supervisor import verify_multilayer_box_supervised


def affine(w,b):
    m=nn.Linear(len(w[0]),len(w),dtype=torch.float64)
    with torch.no_grad():
        m.weight.copy_(torch.tensor(w,dtype=torch.float64))
        m.bias.copy_(torch.tensor(b,dtype=torch.float64))
    return m


def model(unsafe=False):
    first=RoutedLayer(affine([[1.],[-1.]],[0.,0.]),[affine([[1.]],[0.]),affine([[-1.]],[0.])])
    second=RoutedLayer(affine([[1.],[-1.]],[-.5,.5]),
        [affine([[-1.]] if unsafe else [[1.]],[.25] if unsafe else [1.]),affine([[-1.]],[2.])])
    return nn.Sequential(first,second,affine([[1.],[0.]],[0.,0.])).eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--archive',required=True,type=Path)
    args=p.parse_args()
    for target in [args.output_dir,args.archive]:
        if not target.resolve().is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('output scope')
        if target.exists(): raise FileExistsError(target)
    args.output_dir.mkdir(parents=True)
    x=torch.zeros(1,1,dtype=torch.float64)
    cases=[]
    began=time.monotonic()
    for name,unsafe,cap,expected in [('safe',False,4,'POSITIVE'),('unsafe',True,4,'UNSAFE_REPLAYED'),('cap',False,3,'UNKNOWN')]:
        path=args.output_dir/name
        terminal=verify_multilayer_box_supervised(model(unsafe),output_dir=path,total_seconds=300.,max_histories=cap,
            center=x,lower=x-1,upper=x+1,rows=torch.tensor([[1.,-1.]],dtype=torch.float64),thresholds=torch.zeros(1,dtype=torch.float64))
        package=json.loads((path/'candidate.json').read_text())
        result=package['result']
        cases.append({'name':name,'expected_status':expected,'matches_expected':terminal['status']==expected,
            'terminal':terminal,'result':result,'candidate_sha256':hashlib.sha256((path/'candidate.json').read_bytes()).hexdigest()})
    report={'schema':'multilayer-domain-control-archive-v1','cases':cases,'seconds':time.monotonic()-began,
        'status':'PASS' if all(c['matches_expected'] and c['terminal']['audit']['status']=='PASS' for c in cases) else 'FAIL',
        'scope':'analytic two-layer controls, NOT trained-model performance or source-complete certification',
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),ROOT/'act/back_end/moe/multilayer.py',ROOT/'act/back_end/moe/multilayer_audit.py',ROOT/'act/back_end/moe/multilayer_supervisor.py']}}
    args.archive.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':[(c['name'],c['terminal']['status']) for c in cases],'seconds':report['seconds']}))
    if report['status']!='PASS': raise SystemExit(1)


if __name__=='__main__': main()
