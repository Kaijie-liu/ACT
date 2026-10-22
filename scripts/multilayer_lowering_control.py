"""One frozen <=300s ACT dispatch/round-trip and continuation control batch.

Synthetic weights only, no solver, training checkpoint, data, or GPU.
"""
import time
ENTRY=time.monotonic()
import argparse
import ast
import json
from pathlib import Path
import resource
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
from genbab_cpu_ready import ACT

OLD=Path('/data1/Kane/MOE/executions/multilayer_stage1_20261022_r3/act/pipeline/verification/torch2act.py')
CODE=('scripts/multilayer_lowering_control.py','act/pipeline/verification/torch2act.py',
      'act/pipeline/verification/act2torch.py','act/back_end/moe/multilayer.py',
      'act/pipeline/moe/multilayer_training.py','scripts/multilayer_stage1.py',
      'scripts/multilayer_stage1_audit.py','tests/test_multilayer_resume.py')


def dispatch_control(torch):
    from types import SimpleNamespace
    from act.pipeline.verification.torch2act import _LayerGraphBuilder
    class Cat:
        __name__='cat'
        def __repr__(self):return '<built-in method cat of type object at 0xadd123>'
        def __call__(self,*args,**kwargs):return torch.cat(*args,**kwargs)
    graph=torch.fx.Graph();a=graph.placeholder('a');b=graph.placeholder('b')
    node=SimpleNamespace(name='cat',target=Cat(),args=([a,b],),kwargs={'dim':1})
    def builder():
        obj=_LayerGraphBuilder(torch.nn.Identity(),(1,2),dtype=torch.float64)
        obj.node_outputs={'a':[0,1],'b':[2,3]};obj.node_shapes={'a':(1,2),'b':(1,2)}
        obj.next_var=4;return obj
    # Execute only the historical dispatch function on a fixed fake repr;
    # the old process's actual ASLR address was not captured.
    tree=ast.parse(OLD.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_LayerGraphBuilder')
    fun=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_handle_call_function')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fun],type_ignores=[])
    scope={};exec(compile(ast.fix_missing_locations(module),str(OLD),'exec'),scope)
    old=builder();scope['_handle_call_function'](old,node)
    assert 'cat' not in old.node_outputs
    new=builder();new._handle_call_function(node)
    layer=new.layers[-1]
    assert layer.kind=='CONCAT' and layer.in_vars==[0,1,2,3] and len(layer.out_vars)==4
    assert new.node_shapes['cat']==(1,4)
    vals=torch.tensor([[-2.,.25,7.,-3.]],dtype=torch.float64)
    torch.testing.assert_close(vals[:,layer.in_vars],node.target([vals[:,:2],vals[:,2:]],dim=1),rtol=0,atol=0)
    class NoName:
        def __repr__(self):return '<cat at 0xadd123>'
        def __call__(self,*args):return args
    node.target=NoName()
    try:builder()._handle_call_function(node)
    except NotImplementedError:pass
    else:raise AssertionError('missing callable name must not dispatch from repr')
    return dict(old_deterministic_missing_cat=True,new_concat_complete=True,unnamed_rejected=True,
                historical_address_observed=False)


def worker(folder):
    request=load(folder/'request.json');freeze=load(folder.parent/'freeze.json',request['freeze_sha256'])
    assert all(sha(ROOT/p)==h for p,h in freeze['sources'].items()) and sha(OLD)==freeze['old_dispatch_sha256']
    assert resource.getrlimit(resource.RLIMIT_AS)==(8*2**30,8*2**30)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from act.util.device_manager import initialize_device
    initialize_device('cpu','float64')
    dispatch=dispatch_control(torch);save(folder/'dispatch.json',dispatch)
    from act.pipeline.moe.multilayer_training import build
    from act.back_end.moe.multilayer import HistoryPlan,_lower
    from act.pipeline.verification.act2torch import ACTToTorch
    cases=[]
    for depth in (2,3):
        config=dict(input_dim=3072,hidden_output=128,classes=10,router_hidden=[128],expert_hidden=[256,128],layers=depth,experts=8)
        model=build(config,20261022+depth).double().eval()
        center=torch.zeros(1,3072,dtype=torch.float64);plan=HistoryPlan(model,center,weighted_matmul=True)
        for pair in ((0,1),(6,7)):
            history=(pair,)*depth;compiled,score_rows=plan.compile(history)
            width=10+8*depth;net=_lower(compiled,center,center-.5,center+.5,width)
            restored=ACTToTorch(net).run().double().eval()
            actual=[];maximum=0.
            with torch.no_grad():
                for value in (-.5,0.,.5):
                    x=torch.full_like(center,value);expected=compiled(x);got=restored(x)['output']
                    assert expected.shape==got.shape==(1,width)
                    torch.testing.assert_close(got,expected,rtol=1e-10,atol=1e-11)
                    maximum=max(maximum,float((got-expected).abs().max()));actual.append(got)
            assert not torch.equal(actual[0],actual[2]),'dynamic input was folded into center'
            cases.append(dict(depth=depth,pair=pair,outputs=width,router_rows=score_rows,max_error=maximum,
                              matmul_layers=sum(l.kind=='MATMUL' for l in net.layers)))
            save(folder/f'roundtrip_{depth}_{pair[0]}.json',cases[-1])
    import importlib.util
    spec=importlib.util.spec_from_file_location('resume_controls',ROOT/'tests/test_multilayer_resume.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    tests.ResumeControls('test_second_continuation_and_cumulative_binding').test_second_continuation_and_cumulative_binding()
    assert time.monotonic()<request['deadline']
    save(folder/'outcome.json',dict(status='PASS',dispatch=dispatch,roundtrips=cases,
        chain_control='r1/r3/synthetic-r4 plus duplicate/gap/cost/ancestor mutations',
        real_models=0,solver_calls=0,as_limit=resource.getrlimit(resource.RLIMIT_AS),seconds=time.monotonic()-ENTRY))


def run(folder,prior=None):
    assert not folder.exists();folder.mkdir();child=folder/'fixed';child.mkdir()
    retained=None;prior_seconds=0.;prior_bytes=0
    if prior is not None:
        previous=load(prior);prior_seconds=previous['seconds']
        prior_bytes=sum(p.stat().st_size for p in prior.parent.rglob('*') if p.is_file())
        retained=dict(path=str(prior),sha256=sha(prior),seconds=prior_seconds,
                      qualification='numerical controls passed; AS limit not enforced; not sufficient for admission')
    assert resource.getrlimit(resource.RLIMIT_AS)==(8*2**30,8*2**30)
    save(folder/'freeze.json',dict(sources={p:sha(ROOT/p) for p in CODE},old_dispatch_sha256=sha(OLD),
        seconds=300,retained_prior=retained,artifacts=128*2**20,as_bytes=8*2**30,
        controller_as_limit=resource.getrlimit(resource.RLIMIT_AS),threads=1,real_models=0,solver_calls=0,
        cases=['repr address differential/unknown reject','E8 L2/L3 full-width two-history roundtrip','three-root continuation mutations']))
    save(child/'request.json',dict(freeze_sha256=sha(folder/'freeze.json'),deadline=ENTRY+290-prior_seconds))
    env=dict(PATH='/usr/bin:/bin',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    receipt=execute([ACT,'-I','-B',__file__,'worker',str(child)],run_deadline=ENTRY+294-prior_seconds,
        cleanup_deadline=ENTRY+298-prior_seconds,env=env,rss_limit=None,development_policy=POLICY)
    save(folder/'session.json',receipt)
    from b1_observed_batch import session_ok
    good=session_ok(receipt) and (child/'outcome.json').exists() and time.monotonic()<ENTRY+300-prior_seconds
    assert prior_bytes+sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())<128*2**20
    save(folder/'complete.json',dict(status='PASS' if good else 'FAIL',seconds=prior_seconds+time.monotonic()-ENTRY,
        current_seconds=time.monotonic()-ENTRY,retained_prior=retained,
        freeze_sha256=sha(folder/'freeze.json'),session_sha256=sha(folder/'session.json'),
        outcome_sha256=sha(child/'outcome.json') if good else None))
    print(json.dumps(load(folder/'complete.json')),flush=True)


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('folder',type=Path)
    parser.add_argument('--prior',type=Path)
    args=parser.parse_args()
    if args.mode=='run':run(args.folder,args.prior)
    else:
        try:worker(args.folder)
        except BaseException as exc:
            save(args.folder/'error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
