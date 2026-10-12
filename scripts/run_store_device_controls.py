"""Once-only store-device CPU differential under the existing owned CPU clock."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time

from scoped_proof.io import ROOT, PYTHON, save, load, sha
from scoped_source.hz_request_store import identity, require

REFERENCE=Path('/data1/Kane/MOE/baseline_runs/hz_store_supervision_20261013_r1/implementation/act/back_end/moe/store_endpoints.py')
REFERENCE_SHA='1ad73af7dc2367efd036f1efe3e4afcdb38ff2aafe3be0d27b6e142d3ab6e57d'
CASES=('half','negative','mixed','guarded')
EXPECTED={'half':(5,3),'negative':(6,6),'mixed':(6,3),'guarded':(24,12)}
DESIGN='docs/hz_store_device_design_20261013.md'


def fixture(case):
    if case in ('half','negative'):
        from scripts.hz_store_supervised import fixture as f
        return f(case)
    if case=='mixed':
        from scoped_source.hz_store_source_controls import fixture as f
        return f(case)
    require(case=='guarded','frozen source case')
    from fractions import Fraction as F
    affine=lambda rows:dict(kind='affine',rows=rows)
    return dict(schema='REQUEST_STORE_SOURCE_V1',lower=['-1'],upper=['1'],
        router=[affine([['0',[[0,'1']]],['0',[[0,'-1']]],['0',[]]])],
        experts=[[affine([['1/4',[[0,str(sign)]]]]+
                         [['0',[[0,str(F(sign*k,8))]]] for k in range(1,5)])]
                 for sign in (1,-1)]+[[affine([['1/4',[]]]+[['0',[]] for _ in range(4)])]])


def child(root,case,deadline):
    from unittest.mock import patch
    import torch
    torch.set_num_threads(1)
    from act.back_end.moe import store_endpoints as new
    from scoped_source.hz_store_source import build
    from scoped_source.hz_request_store import Store
    from scoped_source.check_store_endpoints import EndpointWork, SCHEMA, plan, check_directory, evaluate_tile
    require(sha(REFERENCE)==REFERENCE_SHA,'archived CPU reference identity')
    spec=importlib.util.spec_from_file_location('frozen_cpu_store_endpoints',REFERENCE)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    root=Path(root); root.mkdir(exist_ok=False); start=time.monotonic()
    work=EndpointWork(deadline); source=fixture(case); save(root/'source.json',source)
    h,creation=build(root/'store',source,work=work)
    store=Store(root/'store',expected_manifest=h,expected_request=identity(source),work=work)
    contract=dict(schema=SCHEMA,store=h,request=identity(source),mode='checked_source',
        classes=store.templates[0].output.count,label=0,margin='0',
        semantics='selected-softmax-top2',ties='ANY_LEGAL_TOPK')
    save(root/'contract.json',contract); anchor=identity(contract)
    inventory=plan(store,contract,expected_contract=anchor,declaration=source,work=work)
    require((len(inventory.tasks),len(inventory.duties))==EXPECTED[case],'frozen complete inventory')
    reports={}
    with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('physical CUDA forbidden')) as guard:
        for name,module in (('old',old),('new',new)):
            begin=time.monotonic(); arm=root/name; arm.mkdir()
            shutil.copytree(root/'store',arm/'store')
            shutil.copyfile(root/'source.json',arm/'source.json')
            shutil.copyfile(root/'contract.json',arm/'contract.json')
            # New default CPU must remain explicit under global-device pollution.
            with torch.device('meta' if name=='new' else 'cpu'):
                proof,stats=module.produce(arm/'proof',store,inventory,tile_width=8,work=work)
            result=check_directory(arm,expected_contract=anchor,expected_proof=proof,work=work)
            reports[name]=dict(proof=proof,result=result,candidate=stats,seconds=time.monotonic()-begin)
        guard.assert_not_called()
    require(reports['old']['proof']==reports['new']['proof'],'all candidate bytes identical')
    require(reports['old']['result']==reports['new']['result'],'all exact complete results identical')
    # Independent original-row residual comparison, not merely outcome counts.
    residuals=[]
    for i,tasks in enumerate(inventory.tiles(8)):
        both=[]
        for name in ('old','new'):
            records=load(root/name/'proof'/f'{i:05d}.json')
            bounds,res=evaluate_tile(store,tasks,[r['candidate'] for r in records],work)
            both.append(dict(bounds=list(map(str,bounds)),residuals=[
                [[j,str(v)] for j,v in sorted(row.items())] for row in res]))
        require(both[0]==both[1],'exact shared residual differential')
        residuals.append(identity(both[0]))
    save(root/'result.json',dict(case=case,source=identity(source),store=h,contract=anchor,
        arms=reports,residual_hashes=residuals,counts=work.counts,source_costs=creation,
        seconds=time.monotonic()-start,physical_cuda=False))
    work.tick()


def parent(root):
    from scripts.hz_store_supervised import sources
    from scoped_proof.owned_bounded import execute
    root=Path(root); root.mkdir(exist_ok=False)
    names=set(sources())|{DESIGN,'scripts/run_store_device_controls.py','scripts/test_store_device.py',
        'scripts/audit_store_device_controls.py','act/back_end/moe/store_device.py'}
    hashes={name:sha(ROOT/name) for name in sorted(names)}
    for name in hashes:
        dest=root/'implementation'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    require(sha(REFERENCE)==REFERENCE_SHA,'fixed CPU reference')
    shutil.copyfile(REFERENCE,root/'reference.py')
    freeze=dict(schema='STORE_DEVICE_CPU_CONTROLS_V1',cases=list(CASES),expected=EXPECTED,
        reference=REFERENCE_SHA,sources=hashes,design=hashes[DESIGN],budget_seconds=300,
        physical_cuda=False,sources_declared={c:identity(fixture(c)) for c in CASES})
    save(root/'freeze.json',freeze)
    calls=[]
    for case in ('unit',*CASES):
        begin=time.monotonic(); deadline=begin+300
        cmd=([PYTHON,'-B','-m','scripts.test_store_device','--report',str(root/'units.json')]
             if case=='unit' else [PYTHON,'-B','-m','scripts.run_store_device_controls',
                 '--child',case,'--output',str(root/case),'--deadline',str(deadline-2)])
        out=execute(cmd,root/f'{case}.log',run_deadline=deadline-2,cleanup_deadline=deadline-1,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),rss_limit=2*2**30)
        path=root/'units.json' if case=='unit' else root/case/'result.json'
        row=dict(case=case,execution=out,receipt=sha(path) if path.is_file() else None,
                 launch_to_receipt_seconds=time.monotonic()-begin,budget_seconds=300)
        terminal=save(root/f'{case}.terminal.json',row)
        observed=time.monotonic()-begin
        save(root/f'{case}.observed.json',dict(terminal_sha256=terminal['sha256'],
            publication_observed_seconds=observed,budget_seconds=300))
        calls.append(row)
        print(case,out['status'],row['launch_to_receipt_seconds'],flush=True)
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if out['status']!='COMPLETED' or observed>=300 or size>64*2**20: break
    save(root/'calls.json',calls)
    require(len(calls)==5 and all(r['execution']['status']=='COMPLETED' for r in calls),'complete frozen roster')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',choices=CASES); p.add_argument('--deadline',type=float); a=p.parse_args()
    if a.child: child(a.output,a.child,a.deadline)
    else: parent(a.output)
