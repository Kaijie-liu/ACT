"""Five frozen synthetic endpoint calls, each with one parent-owned deadline."""
import argparse
from fractions import Fraction as F
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO))
from scoped_source.hz_request_store import compact, digest, identity, Store, require
from scoped_source.check_store_endpoints import SCHEMA, EndpointWork, plan, check_directory

CASES=('source8_tile8','source8_tile1','negative3_tile8','mixed_tile8','given_large_tile8')
FILES=('scoped_source/hz_request_store.py','scoped_source/hz_store_write.py',
       'scoped_source/check_hz_store_source.py','scoped_source/hz_store_source.py',
       'scoped_source/hz_store_sparse.py','scoped_source/hz_store_source_controls.py',
       'scoped_source/hz_store_control_build.py','scoped_source/check_store_endpoints.py',
       'act/back_end/moe/store_endpoints.py','act/back_end/solver/solver_hz.py',
       'scripts/test_store_endpoints.py','scripts/check_store_endpoints.py',
       'scripts/audit_store_endpoints.py','scripts/test_store_endpoints_units.py',
       'docs/hz_store_endpoints_design_20261013.md')


def save(path,value):
    with Path(path).open('xb') as stream: stream.write(compact(value))


def fixture(case):
    if case=='mixed_tile8':
        from scoped_source.hz_store_source_controls import fixture as old
        return old('mixed')
    n=3 if case=='negative3_tile8' else 8
    affine=lambda rows: dict(kind='affine',rows=rows)
    router=affine([['0',[] if n==3 else [[i,'1']]] for i in range(n)])
    experts=[]
    for i in range(n):
        outputs=[['2',[[0,'1']]]]+[['0',[[0,str(F(k,32))]]] for k in range(1,3 if n==3 else 10)]
        if n==3: outputs[-1]=['3',[[0,'1']]]
        experts.append([affine([['0',[[i,'1']]]]),{'kind':'relu'},affine(outputs)])
    return dict(schema='REQUEST_STORE_SOURCE_V1',lower=['-1']*n,upper=['1']*n,router=[router],experts=experts)


def child(case,root,deadline):
    import torch
    torch.set_num_threads(1)
    from act.back_end.moe.store_endpoints import produce
    from scoped_source.hz_store_source import build
    start=time.monotonic(); work=EndpointWork(deadline); root=Path(root); root.mkdir(exist_ok=False)
    before=time.monotonic(); declaration=None
    if case=='given_large_tile8':
        from scoped_source.hz_store_control_build import build as given
        binding=given(root/'store',large=True,work=work); h=binding['manifest']; request=binding['request']
        source_cost={}; mode='given_hz'
    else:
        declaration=fixture(case); request=identity(declaration)
        save(root/'source.json',declaration); h,source_cost=build(root/'store',declaration,work=work)
        mode='checked_source'
    creation=time.monotonic()-before; before=time.monotonic()
    store=Store(root/'store',expected_manifest=h,expected_request=request,work=work)
    c=store.templates[0].output.count
    contract=dict(schema=SCHEMA,store=h,request=request,mode=mode,classes=c,label=0,margin='0',
                  semantics='selected-softmax-top2',ties='ANY_LEGAL_TOPK')
    save(root/'contract.json',contract); anchor=identity(contract)
    inventory=plan(store,contract,expected_contract=anchor,declaration=declaration,work=work)
    preparation=time.monotonic()-before; before=time.monotonic()
    width=1 if case.endswith('tile1') else 8
    proof,stats=produce(root/'proof',store,inventory,tile_width=width,work=work)
    generation=time.monotonic()-before; before=time.monotonic()
    result=check_directory(root,expected_contract=anchor,expected_proof=proof,work=work)
    checking=time.monotonic()-before; work.tick()
    report=dict(case=case,contract=anchor,proof=proof,store=h,request=request,result=result,
        counts=work.counts,store_stats=dict(store.stats),candidate=stats,source_cost=source_cost,
        costs=dict(child_after_import_seconds=time.monotonic()-start,source_creation_seconds=creation,
            store_and_source_check_seconds=preparation,proposal_and_evidence_seconds=generation,
            independent_reception_seconds=checking,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        query_ids=[t.query_id for t in inventory.tasks],slots=[t.slot for t in inventory.tasks],
        gates=[dict(pair=list(p),bounds=list(map(str,g))) for p,g in inventory.gates])
    save(root/'result.json',report); work.tick(); print(compact(dict(case=case,result=digest(compact(report)))).decode())


def parent(root):
    root=Path(root); require(not root.exists(), 'new attempt directory required'); root.mkdir(parents=True)
    code=root/'code'; code.mkdir(); hashes={}
    for name in FILES:
        raw=(REPO/name).read_bytes(); hashes[name]=digest(raw); target=code/name
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(REPO/name,target)
    save(root/'source_contract.json',dict(files=hashes,protocol=hashes[FILES[-1]],
        python=sys.executable,python_version=sys.version,cases=list(CASES),budget_seconds=300,
        max_visits=96000000,archive_cap_bytes=128*2**20))
    rows=[]; halted=False
    for case in CASES:
        if halted:
            rows.append(dict(case=case,status='NOT_RUN_AFTER_FAILURE')); continue
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        require(size<127*2**20, 'stage archive reserve')
        started=time.monotonic(); deadline=started+300
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--child',case,'--output',str(root/case),'--deadline',str(deadline)]
        with (root/f'{case}.stdout').open('xb') as stdout, (root/f'{case}.stderr').open('xb') as stderr:
            p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            try:
                p.wait(timeout=max(.001,deadline-time.monotonic())); status='COMPLETE' if p.returncode==0 else 'ERROR'
                if time.monotonic()>deadline: status='TIMEOUT'
            except subprocess.TimeoutExpired:
                p.kill()
                try: p.wait(timeout=2)
                except subprocess.TimeoutExpired: pass
                status='TIMEOUT'
        elapsed=time.monotonic()-started; path=root/case/'result.json'
        row=dict(case=case,status=status,returncode=p.returncode,launch_to_receipt_seconds=elapsed,
                 result_sha256=digest(path.read_bytes()) if path.is_file() else None)
        rows.append(row); save(root/f'{case}.terminal.json',row)
        print(json.dumps(row),flush=True); halted=status!='COMPLETE'
    payload=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    save(root/'terminals.json',dict(calls=rows,payload_bytes_before_ledger=payload))
    require(payload<=128*2**20, 'stage archive byte cap')
    return 1 if halted else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--child',choices=CASES); p.add_argument('--deadline',type=float); a=p.parse_args()
    if a.child:
        try: child(a.child,a.output,a.deadline)
        except BaseException:
            traceback.print_exc(); sys.exit(1)
    else: sys.exit(parent(a.output))
