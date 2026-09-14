"""Separate real-model static obligations on a pinned external CROWN frontend."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from act.pipeline.moe.external_compatibility import ROOT, TOOL, ENV, COMMITS, git, sha, dump

PAIRS=((2,4),(4,5))
PARENT=ROOT/'data/moe/results/request_lp_20260914_r1'
PARENT_HASH='90aad930c9f8b1f7018c3a7ef30cee80ab4487e5fae16b231bc6432c7b3cf082'
TENSOR_HASH='2523f7b04070af1eb4ab5d8bef282828902abfa921d3128f5ff3b9206916867e'


def request_identity():
    if sha(PARENT/'manifest.json')!=PARENT_HASH or sha(PARENT/'request.pt')!=TENSOR_HASH:
        raise ValueError('parent request drift')
    manifest=json.loads((PARENT/'manifest.json').read_text())
    if manifest['routes']['feasible']!=[list(p) for p in PAIRS] or not manifest['routes']['exact']:
        raise ValueError('static pair inventory differs from parent')
    return manifest['request'],manifest['request_id']


def worker(pair,path):
    start=time.monotonic()
    result={'pair':list(pair),'phase':'imports','status':'INCOMPLETE','formal_SAFE':False,
            'task':'STATIC_VARIABLE_WEIGHT_WHOLE_BOX_SUFFICIENT_OBLIGATION','python':sys.version}
    try:
        sys.path[:0]=[str(TOOL/'auto_LiRPA'),str(TOOL/'complete_verifier')]
        from act.util.typing_compat import install_typing_override
        install_typing_override()
        import torch
        import auto_LiRPA
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        from act.back_end.moe.factory import load_output_moe_checkpoint
        from act.back_end.moe.static_pair import StaticSelectedSoftmaxPair
        from act.pipeline.moe.staged_verifier import _tensor_identity, _model_state_identity
        from act.pipeline.moe.check_request_lp import property_row
        torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
        result.update(torch=torch.__version__,auto_lirpa_file=auto_LiRPA.__file__,device='cpu',dtype='float64')
        if not Path(auto_LiRPA.__file__).resolve().is_relative_to(TOOL/'auto_LiRPA'):
            raise ValueError('incorrect imported frontend')
        request,rid=request_identity();result.update(request=request,request_id=rid)
        result['phase']='checkpoint_and_input'
        subject=json.loads((ROOT/'act/pipeline/moe/configs/schedule_confirmation_selection_r2.json').read_text())['models']['seed0']
        if sha(Path(subject['checkpoint']))!=request['checkpoint_sha256']:raise ValueError('checkpoint drift')
        model,_=load_output_moe_checkpoint(subject['checkpoint'],map_location='cpu');model.cpu().double().eval()
        state=_model_state_identity(model)
        if state!=subject['model_state']:raise ValueError('model-state mismatch')
        result['model_state']=state
        tensors=torch.load(PARENT/'request.pt',weights_only=True,map_location='cpu')
        for k,t in tensors.items():
            if _tensor_identity(t)!=request[k]:raise ValueError('represented-input mismatch')
        center,lo,hi=(tensors[k] for k in ('center','lower','upper'))
        adapter=StaticSelectedSoftmaxPair(model,list(pair))
        classes=request['classes'];pred=request['clean_prediction']
        C=torch.tensor([property_row(classes,pred,i) for i in range(classes-1)],dtype=torch.float64).unsqueeze(0)
        result['C']=C.tolist()
        mask=(torch.arange(center.numel()).reshape(center.shape)%2==0)
        points=[center,lo,hi,torch.where(mask,lo,hi),torch.where(mask,hi,lo)]
        reference=[];probe_errors=[];weights=[]
        result['phase']='concrete_conformance'
        with torch.no_grad():
            if int(model(center).argmax())!=pred:raise ValueError('clean prediction differs')
            for point in points:
                scores=model.router(point)[:,list(pair)]
                w=torch.softmax(scores,dim=1)
                outputs=torch.stack([model.experts[i](point) for i in pair],dim=1)
                forced=(w.unsqueeze(-1)*outputs).sum(dim=1)
                probe_errors.append(float((adapter(point)-forced).abs().max()))
                reference.append(forced);weights.append(w.tolist())
        result['forced_branch_max_error']=max(probe_errors);result['probe_weights']=weights
        result['concrete_margins']=[(C@output.unsqueeze(-1)).squeeze(-1).tolist()[0] for output in reference]
        if max(probe_errors)>1e-10:raise ValueError('static expression conformance mismatch')
        result['phase']='graph_conversion';phase_start=time.monotonic()
        bounded=BoundedModule(adapter,center,device='cpu',bound_opts={'conv_mode':'matrix'})
        result['graph_seconds']=time.monotonic()-phase_start
        result['nodes']=[{'name':n,'type':type(v).__name__} for n,v in bounded._modules.items()]
        result['phase']='lowered_conformance'
        with torch.no_grad():
            errors=[float((bounded(point)-expected).abs().max()) for point,expected in zip(points,reference)]
        result['lowered_max_error']=max(errors)
        if max(errors)>1e-10:raise ValueError('lowered graph conformance mismatch')
        result['phase']='plain_CROWN';phase_start=time.monotonic()
        bounded_input=BoundedTensor(center,PerturbationLpNorm(norm=float('inf'),x_L=lo,x_U=hi))
        lower,upper=bounded.compute_bounds(x=(bounded_input,),C=C,method='CROWN')
        result['bound_seconds']=time.monotonic()-phase_start
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():raise ValueError('nonfinite bounds')
        result['bounds']={'lower':lower.detach().tolist()[0],'upper':upper.detach().tolist()[0]}
        result['positive_rows']=int((lower>1e-7).sum());result['required_rows']=classes-1
        result.update(status='NUMERICAL_BOUND_RETURNED_NOT_FORMAL_SAFE',phase='complete')
    except Exception as exc:
        result.update(status='ERROR_OR_UNSUPPORTED',exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-start;dump(path,result)


def run(root):
    if git(ROOT,'branch','--show-current')!='feat/moe-route-verification' or git(ROOT,'status','--porcelain'):
        raise RuntimeError('clean feature branch required')
    root=root.resolve()
    if not root.is_relative_to(ROOT/'data/moe/results'):raise ValueError('invalid result root')
    for path,commit in COMMITS.items():
        if git(path,'rev-parse','HEAD')!=commit or git(path,'status','--porcelain'):raise ValueError('tool source drift')
    request,rid=request_identity();root.mkdir(exist_ok=False)
    launch={'execution_head':git(ROOT,'rev-parse','HEAD'),'commits':COMMITS,'environment':ENV,
            'worker_sha256':sha(Path(__file__)),'adapter_sha256':sha(ROOT/'act/back_end/moe/static_pair.py'),
            'parent_manifest_sha256':PARENT_HASH,'request_tensor_sha256':TENSOR_HASH,'request':request,'request_id':rid,
            'pairs':PAIRS,'timeout_each_seconds':120,'config':{'method':'CROWN','bound_opts':{'conv_mode':'matrix'},'dtype':'float64','device':'cpu'},
            'scope':'Two whole-box static variable-weight obligations; not a complete dynamic-model benchmark or strict proof.'}
    dump(root/'launch.json',launch)
    for pair in PAIRS:
        name=f'pair_{pair[0]}_{pair[1]}';path=root/(name+'.json')
        with (root/(name+'.log')).open('w') as log:
            try:
                done=subprocess.run([ENV,'-m','act.pipeline.moe.external_static_pair','--worker',str(pair[0]),str(pair[1]),'--output',str(path)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=False,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
                if not path.exists():dump(path,{'pair':list(pair),'status':'WORKER_FAILED','returncode':done.returncode,'formal_SAFE':False})
            except subprocess.TimeoutExpired:dump(path,{'pair':list(pair),'status':'TIMEOUT','formal_SAFE':False})
    results=[json.loads((root/f'pair_{a}_{b}.json').read_text()) for a,b in PAIRS]
    all_positive=all(r.get('positive_rows')==9 and r.get('status')=='NUMERICAL_BOUND_RETURNED_NOT_FORMAL_SAFE' for r in results)
    summary={'launch':launch,'results':results,'required_rows':18,'positive_rows':sum(r.get('positive_rows',0) for r in results),
             'status':'ALL_LISTED_STATIC_OBLIGATIONS_NUMERICALLY_POSITIVE' if all_positive else 'NOT_ALL_STATIC_OBLIGATIONS_POSITIVE',
             'formal_SAFE':False,'raw_hashes':{p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}}
    dump(root/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'pairs':[{k:r[k] for k in ('pair','phase','status','positive_rows','message') if k in r} for r in results]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=int,nargs=2);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.worker:
        if tuple(args.worker) not in PAIRS:raise ValueError('unregistered pair')
        worker(tuple(args.worker),args.output)
    else:run(args.output)
