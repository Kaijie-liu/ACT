"""Three frozen external frontend probes; no performance or formal SAFE claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/data1/Kane/MOE/ACT')
TOOL=Path('/data1/Kane/MOE/baselines/alpha-beta-CROWN')
ENV='/data1/Kane/MOE/envs/alpha-beta-crown/bin/python'
COMMITS={str(TOOL):'e5c7e17bf0488843acb77b7519f59876717a49f4',
         str(TOOL/'auto_LiRPA'):'5a098e8f9fb5786a428a024981d833d303921f2d'}
CASES=('dynamic_top2','static_pair','input_polytope')


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_model(case):
    import torch
    if case not in CASES[:2]: raise ValueError('not a toy graph case')
    class Toy(torch.nn.Module):
        def forward(self,x):
            a,b=x[:,0:1],x[:,1:2]
            scores=torch.cat((a,b,-a),dim=1)
            experts=torch.cat((1+a,1-b,1+b),dim=1)
            if case=='dynamic_top2':
                values,indices=torch.topk(scores,2,dim=1)
                output=torch.gather(experts,1,indices)
            else: values=scores[:,:2];output=experts[:,:2]
            return (torch.softmax(values,dim=1)*output).sum(dim=1,keepdim=True)
    return Toy().eval()


def worker(case,path):
    result={'case':case,'phase':'imports','status':'INCOMPLETE','python':sys.version,'formal_SAFE':False}
    start=time.monotonic()
    try:
        sys.path[:0]=[str(TOOL/'auto_LiRPA'),str(TOOL/'complete_verifier')]
        import torch
        import auto_LiRPA
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        torch.set_num_threads(1)
        result.update(torch=torch.__version__,auto_lirpa_file=auto_LiRPA.__file__,dtype='float32',device='cpu')
        if not Path(auto_LiRPA.__file__).resolve().is_relative_to(TOOL/'auto_LiRPA'):
            raise RuntimeError('wrong auto_LiRPA import provenance')
        if case=='input_polytope':
            result['phase']='full_api_import'
            import api
            result['api_file']=api.__file__; result['phase']='box_control'
            x=api.input_vars(2); box=(x>=-.1)&(x<=.1)
            lo,hi=api._parse_input_bounds(box,x)
            result['box']={'lower':lo.tolist(),'upper':hi.tolist()}
            result['phase']='relational_input_parse'
            lo,hi=api._parse_input_bounds(box & (x[0]+x[1]<=.1),x)
            result['polytope_return']={'lower':lo.tolist(),'upper':hi.tolist()}
            result['status']='PARSER_ACCEPTED_REQUIRES_SEMANTIC_REVIEW'
        else:
            model=toy_model(case);center=torch.zeros(1,2);lo=center-.1;hi=center+.1
            grid=torch.cartesian_prod(torch.linspace(-.1,.1,5),torch.linspace(-.1,.1,5))
            concrete=model(grid).detach()
            result['grid_range']=[float(concrete.min()),float(concrete.max())]
            result['phase']='graph_conversion'
            bounded=BoundedModule(model,center,device='cpu',bound_opts={'conv_mode':'matrix'})
            result['nodes']=[{'name':name,'type':type(node).__name__} for name,node in bounded._modules.items()]
            differences=[]
            for point in grid:
                differences.append(float((bounded(point.unsqueeze(0))-model(point.unsqueeze(0))).abs().max()))
            result['max_finite_probe_error']=max(differences)
            result['phase']='plain_CROWN'
            inp=BoundedTensor(center,PerturbationLpNorm(norm=float('inf'),x_L=lo,x_U=hi))
            lower,upper=bounded.compute_bounds(x=(inp,),method='CROWN')
            result['bounds']={'lower':lower.detach().tolist(),'upper':upper.detach().tolist()}
            result['status']='NUMERICAL_BOUND_RETURNED_NOT_FORMAL_SAFE'
        result['phase']='complete'
    except Exception as exc:
        result.update(status='REJECTED_OR_ERROR',exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-start;dump(path,result)


def run(output):
    if git(ROOT,'branch','--show-current')!='feat/moe-route-verification' or git(ROOT,'status','--porcelain'):
        raise RuntimeError('clean feature branch required')
    output=output.resolve()
    if not output.is_relative_to(ROOT/'data/moe/results'): raise ValueError('invalid result root')
    for path,commit in COMMITS.items():
        if git(path,'rev-parse','HEAD')!=commit or git(path,'status','--porcelain'):
            raise RuntimeError('external source drift')
    output.mkdir(exist_ok=False)
    anchors=['auto_LiRPA/auto_LiRPA/operators/indexing.py','auto_LiRPA/auto_LiRPA/operators/softmax.py',
             'auto_LiRPA/auto_LiRPA/bound_op_map.py','complete_verifier/api.py']
    launch={'execution_head':git(ROOT,'rev-parse','HEAD'),'commits':COMMITS,'environment':ENV,
            'worker_sha256':sha(Path(__file__)),'anchors':{a:sha(TOOL/a) for a in anchors},
            'cases':CASES,'timeout_each_seconds':120,'scope':'Frontend compatibility only; NOT complete-tool performance.'}
    dump(output/'launch.json',launch)
    for case in CASES:
        path=output/(case+'.json');start=time.monotonic()
        with (output/(case+'.log')).open('w') as log:
            try:
                done=subprocess.run([ENV,str(Path(__file__).resolve()),'--worker',case,'--output',str(path)],
                    stdout=log,stderr=subprocess.STDOUT,timeout=120,check=False,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
                if not path.exists(): dump(path,{'case':case,'status':'WORKER_FAILED','returncode':done.returncode})
            except subprocess.TimeoutExpired:
                dump(path,{'case':case,'status':'TIMEOUT','seconds':time.monotonic()-start})
    results={case:json.loads((output/(case+'.json')).read_text()) for case in CASES}
    dump(output/'summary.json',{'launch':launch,'results':results,'raw_hashes':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({k:{t:v[t] for t in ('status','phase','message') if t in v} for k,v in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',choices=CASES)
    args=p.parse_args()
    if args.worker:worker(args.worker,args.output)
    else:run(args.output)
