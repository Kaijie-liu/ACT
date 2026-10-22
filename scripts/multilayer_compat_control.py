"""Fifth disclosed compatibility edit: sparse multiplication specs.

One <=300s synthetic batch; no checkpoint/data/GPU and no rerun of stage0.
Full-width semantic cases and operator differentials are distinct controls.
"""
import time
ENTRY=time.monotonic()
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
import genbab_cpu_ready as backend

CODE=('scripts/multilayer_compat_control.py','scripts/genbab_cpu_ready.py',
      'scripts/b2_observed_worker.py','act/back_end/moe/multilayer_external.py')


def operator_controls(torch):
    from types import SimpleNamespace
    from auto_LiRPA.operators.bivariate import BoundMul
    torch.manual_seed(20261022)
    x=SimpleNamespace(lower=torch.full((1,1),-.7),upper=torch.full((1,1),.8),
        output_shape=(1,1),batch_dim=0,perturbed=True)
    y=SimpleNamespace(lower=torch.linspace(-2,-.1,128)[None],
        upper=torch.linspace(.1,3,128)[None],output_shape=(1,128),batch_dim=0,perturbed=True)
    node=BoundMul(inputs=[],options={});node._start='s';node.opt_stage='opt'
    alpha=torch.rand((4,256,1,128),requires_grad=True);node.alpha={'s':alpha}
    a=torch.randn(256,1,128);u=torch.randn_like(a)
    indices=torch.arange(1,249,4,dtype=torch.long)  # 62 nonconsecutive specs
    full=node.bound_backward_both_perturbed(a,u,x,y)
    sparse=node.bound_backward_both_perturbed(a[indices],u[indices],x,y,unstable_idx=indices)
    def flat(value):
        if isinstance(value,(list,tuple)):
            return [v for item in value for v in flat(item)]
        return [] if value is None else [value]
    for f,s in zip(flat(full),flat(sparse)):
        torch.testing.assert_close(f[indices],s,rtol=0,atol=0)
    sum(v.sum() for v in flat(sparse)).backward()
    other=torch.ones(256,dtype=torch.bool);other[indices]=False
    assert torch.count_nonzero(alpha.grad[:,other])==0
    assert torch.count_nonzero(alpha.grad[:,indices])>0
    # Shared alpha remains broadcast; no spec truncation or duplicate indexing.
    node.alpha={'s':torch.rand((4,1,1,128))}
    shared=node.bound_backward_both_perturbed(a,u,x,y)
    ss=node.bound_backward_both_perturbed(a[indices],u[indices],x,y,unstable_idx=indices)
    for f,s in zip(flat(shared),flat(ss)):torch.testing.assert_close(f[indices],s,rtol=0,atol=0)
    node.opt_stage=None
    ordinary=node.bound_backward_both_perturbed(a,u,x,y)
    osparse=node.bound_backward_both_perturbed(a[indices],u[indices],x,y,unstable_idx=indices)
    for f,s in zip(flat(ordinary),flat(osparse)):torch.testing.assert_close(f[indices],s,rtol=0,atol=0)
    return dict(status='PASS',dense_specs=256,sparse_specs=len(indices),features=128,
        differential='all six slope/bias terms, lower/upper, gradients, shared alpha, nonoptimized',
        guarantee='operator identity, not network proof')


def fixture(torch,layers,case):
    from torch import nn
    from act.back_end.moe.multilayer import RoutedLayer
    from act.back_end.moe.multilayer_external import SequentialTop2
    def linear(i,o):
        m=nn.Linear(i,o).double()
        with torch.no_grad():
            m.weight.zero_();m.bias.zero_()
        return m
    modules=[];width=3072
    for depth in range(layers):
        out=10 if depth==layers-1 else 128
        router=nn.Sequential(linear(width,128),nn.ReLU(),linear(128,2))
        scalar_scale=1/4096 if depth==0 else 1/128
        with torch.no_grad():
            router[0].weight.fill_(scalar_scale);router[0].bias.fill_(2)
            router[2].weight[0].fill_(1/128);router[2].weight[1].fill_(-1/128)
            router[2].bias.copy_(torch.tensor([-2.,2.]))
        if case=='tie':
            with torch.no_grad():router[-1].weight.zero_();router[-1].bias.zero_()
        experts=[]
        for expert in range(2):
            seq=nn.Sequential(linear(width,256),nn.ReLU(),
                linear(256,128),nn.ReLU(),linear(128,out))
            with torch.no_grad():
                seq[0].weight.fill_(scalar_scale);seq[0].bias.fill_(2)
                seq[0].bias[torch.arange(1,249,4)]=0  # exactly62 nonconsecutive unstable rows
                seq[2].weight.fill_(1/256)
                if depth==layers-1:
                    seq[-1].weight[0].fill_((1 if expert==0 else -1)/512)
                    seq[-1].bias[0]=-1/4 if case=='unsafe' else 1
                else:seq[-1].weight.fill_((1 if expert==0 else -1)/512)
            experts.append(seq)
        modules.append(RoutedLayer(router,experts,k=2,mode='selected_softmax'));width=out
    model=SequentialTop2(modules).eval()
    center=torch.zeros(1,3072);lower=center-1;upper=center+1
    rows=torch.zeros(9,10);rows[:,0]=1
    for j in range(9):rows[j,j+1]=-1
    return model,dict(center=center,lower=lower,upper=upper,rows=rows,thresholds=torch.zeros(9))


def run_worker(folder):
    request=load(folder/'request.json');deadline=request['deadline']
    backend.limits(deadline)
    log=os.open(folder/'native.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.dup2(log,1);os.dup2(log,2);os.close(log)
    save(folder/'environment.json',backend.probe())
    import torch
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    from act.back_end.moe.multilayer_external import StaticPrefix,GuardedHistory
    from act.pipeline.moe.multilayer_study import affine_interval
    from b2_observed_worker import verify_graph
    from auto_LiRPA import BoundedModule,BoundedTensor,PerturbationLpNorm
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
    save(folder/'operator.json',operator_controls(torch))
    records=[]
    for layers in (2,3):
        for case in ('safe','unsafe','tie'):
            if time.monotonic()>=deadline:raise TimeoutError('control deadline')
            model,t=fixture(torch,layers,case);history=((0,1),)*layers
            static=StaticPrefix(model,history).eval();graph=GuardedHistory(model,history,t['rows'],t['thresholds']).eval()
            # Independent interval induction: every layer's selected mixture
            # is in the coordinatewise hull of BOTH expert intervals.
            lo,hi=t['lower'],t['upper']
            for layer in model.layers:
                bounds=[affine_interval(e,lo,hi) for e in layer.experts]
                lo=torch.stack([v[0] for v in bounds]).amin(0)
                hi=torch.stack([v[1] for v in bounds]).amax(0)
            margin=lo[0,0]-hi[0,1:]
            if case!='unsafe':assert bool((margin>0).all())
            grid=(torch.arange(-128,129)/128)[:,None].expand(-1,3072).clone()
            with torch.no_grad():
                torch.testing.assert_close(model(grid),static(grid),rtol=1e-12,atol=1e-12)
                expected=model(grid)@t['rows'].T
                torch.testing.assert_close(graph(grid),expected,rtol=1e-12,atol=1e-12)
                if case=='unsafe':assert bool((expected[128]<0).all())
            if case=='unsafe':
                record=dict(status='UNSAFE_REPLAYED',center_margin=expected[128].tolist())
            else:
                record,_=verify_graph(graph,t['lower'],t['upper'],9,deadline)
                assert record['status']=='verified', 'analytic wide positive control'
            # Alpha path is explicitly exercised even if API's initial CROWN
            # already closes the large positive calibration margin.
            if case=='safe':
                bounded=BoundedModule(graph,t['center'],device='cpu',bound_opts={
                    'optimize_bound_args':{'iteration':2,'enable_alpha_crown':True}})
                x=BoundedTensor(t['center'],PerturbationLpNorm(norm=float('inf'),x_L=t['lower'],x_U=t['upper']))
                lb,_=bounded.compute_bounds(x=(x,),method='CROWN-Optimized',bound_upper=False)
                assert bool(torch.isfinite(lb).all()) and bool((lb>0).all())
                record['forced_alpha_minimum']=float(lb.min())
            row=dict(layers=layers,case=case,history_count=1,classes=10,properties=9,
                input=3072,expert_hidden=[256,128],dimensions_only_not_E8_capacity=True,result=record)
            records.append(row);save(folder/f'case_{layers}_{case}.json',row)
    if time.monotonic()>=deadline:raise TimeoutError('control reception deadline')
    save(folder/'outcome.json',dict(status='PASS',cases=records,operator=load(folder/'operator.json'),
        seconds=time.monotonic()-ENTRY,trained_models=0,gpu=0))


def main(folder):
    assert not folder.exists();folder.mkdir();child=folder/'fixed';child.mkdir()
    manifest=dict(code={p:sha(ROOT/p) for p in CODE},backend=backend.source_identity(),
        cases=['L2/L3 full-width safe/unsafe/tie'],seconds=300,per_process_AS=8*2**30,
        numerical_config=backend.CONFIG,real_models=0,stop='any incorrect semantic or differential')
    save(folder/'freeze.json',manifest)
    save(child/'request.json',dict(deadline=ENTRY+290,freeze_sha256=sha(folder/'freeze.json')))
    env=dict(PATH='/usr/bin:/bin',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    receipt=execute([str(backend.ENVIRONMENT/'bin/python'),'-I','-B',__file__,'worker',str(child)],
        run_deadline=ENTRY+294,cleanup_deadline=ENTRY+298,env=env,rss_limit=None,development_policy=POLICY)
    save(folder/'session.json',receipt)
    from b1_observed_batch import session_ok
    ok=session_ok(receipt) and (child/'outcome.json').exists() and time.monotonic()<ENTRY+300
    save(folder/'complete.json',dict(status='PASS' if ok else 'FAIL',seconds=time.monotonic()-ENTRY,
        freeze_sha256=sha(folder/'freeze.json'),session_sha256=sha(folder/'session.json'),
        outcome_sha256=sha(child/'outcome.json') if ok else None))
    print(json.dumps(load(folder/'complete.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('path',type=Path)
    a=p.parse_args()
    if a.mode=='run':main(a.path)
    else:
        try:run_worker(a.path)
        except BaseException as exc:
            save(a.path/'error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
