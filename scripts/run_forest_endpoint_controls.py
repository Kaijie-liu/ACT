"""Once-only frozen CPU component calls; not production experiment admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
DESIGN='docs/hz_forest_endpoints_design_20261013.md'
CASES=('small40','mixed','guarded_w8','half','guarded_w1')
ANCHORS={'small40':'28c41d0da9493f2b80de1dab910a769868f12240ac00dc55727b9a716dc8fe65',
 'mixed':'24155a8113346d6964b6a4497be0abc0e5dcebbb169c949931468d6fdbb92a22',
 'guarded_w8':'0be644fc20d28ab6d811f4c933f58456d9039e7a1d513e38d41f929b92cfc8da',
 'half':'cf4dbc3dff90da9d71fc9a056e29cdefa69a2aa8dae768f08555a7b84fb4c8a0',
 'guarded_w1':'0be644fc20d28ab6d811f4c933f58456d9039e7a1d513e38d41f929b92cfc8da'}
LIMIT=64*2**20
FILES=('scoped_source/forest_endpoints.py','act/back_end/moe/forest_endpoints.py',
 'scoped_source/forest_endpoint_controls.py','scoped_source/verify_forest_endpoints.py',
 'scoped_source/source_dag.py','scoped_source/check_source_dag.py','scoped_source/source_dag_controls.py',
 'scoped_source/coefficient_dag.py','scoped_source/hz_request_store.py',
 'scoped_source/tensor_rows.py','scoped_source/tensor_row_controls.py','scoped_source/factored_io.py',
 'scoped_source/factored_source.py','source_enclosure/format.py','upstream_source/checker.py',
 'scoped_source/hz_store_source.py','scoped_source/check_hz_store_source.py','scoped_source/hz_store_sparse.py',
 'scoped_source/hz_store_write.py','scoped_source/check_store_endpoints.py',
 'scripts/hz_store_supervised.py','scripts/run_store_device_controls.py','scoped_proof/io.py',
 'scripts/test_forest_endpoints.py','scripts/test_source_dag.py','scripts/test_source_dag_audit.py',
 'scripts/test_coefficient_dag.py','scoped_source/coefficient_dag_controls.py','scripts/test_tensor_rows.py',
 'scripts/audit_source_dag_controls.py','scripts/run_source_dag_controls.py','scripts/run_tensor_rows_controls.py',
 'scripts/run_forest_endpoint_controls.py','scripts/audit_forest_endpoint_controls.py',DESIGN)


def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)


def worker(root,name,deadline):
    start=time.monotonic(); root=Path(root); root.mkdir(exist_ok=False)
    phases={}; receipt=dict(status='INCOMPLETE',name=name); w=None
    try:
        import torch
        torch.set_num_threads(1)
        from scoped_source.hz_request_store import Work,compact,digest,identity,require
        from scoped_source.forest_endpoint_controls import fixture,eager,differential
        from scoped_source.source_dag import build
        from scoped_source.forest_endpoints import CheckedForest,contract,plan,check_proof
        from act.back_end.moe.forest_endpoints import produce
        from unittest.mock import patch
        phases['imports_seconds']=time.monotonic()-start; w=Work(deadline)
        begin=time.monotonic(); bundle=root/'bundle'; bundle.mkdir(); case='guarded' if name.startswith('guarded') else name
        source_root,source=fixture(bundle/'source',case)
        require(source==ANCHORS[name], 'frozen captured source identity')
        if source_root!=bundle/'source':
            # The mixed fixture's initial template is not its declared source.
            shutil.move(str(bundle/'source'),str(root/'fixture_template'))
            shutil.move(str(source_root),str(bundle/'source')); source_root=bundle/'source'
        phases['capture_seconds']=time.monotonic()-begin; begin=time.monotonic()
        raw,generation=build(source_root,source,work=w)
        phases['source_build_encode_seconds']=time.monotonic()-begin
        begin=time.monotonic(); (bundle/'forest.json').write_bytes(raw)
        phases['forest_publish_seconds']=time.monotonic()-begin; begin=time.monotonic()
        f=CheckedForest(raw,source_root,source,expected=digest(raw),work=w)
        phases['source_check_seconds']=time.monotonic()-begin; begin=time.monotonic()
        c=contract(f); p=plan(f,c,expected=identity(c),work=w)
        phases['planning_seconds']=time.monotonic()-begin; begin=time.monotonic()
        partial=root/'partial'; partial.mkdir()
        def event(i,records):
            with (partial/f'{i:05d}.json').open('xb') as s: s.write(compact(records))
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('physical CUDA forbidden')):
            proof,native=produce(f,p,width=1 if name.endswith('w1') else 8,work=w,event=event)
        phases['candidate_and_exact_selection_seconds']=time.monotonic()-begin; begin=time.monotonic()
        for name_,value in (('contract.json',c),('proof.json',proof)):
            with (bundle/name_).open('xb') as s: s.write(compact(value))
        phases['proof_encode_publish_seconds']=time.monotonic()-begin; begin=time.monotonic()
        result=check_proof(f,c,proof,expected_contract=identity(c),expected_proof=identity(proof),work=w)
        phases['complete_endpoint_check_seconds']=time.monotonic()-begin; begin=time.monotonic()
        store,comparison=eager(source_root,source,raw,root/'test_only_eager',w)
        comparison['exact_endpoints']=differential(store,f,p,proof,w)
        phases['test_only_eager_differential_seconds']=time.monotonic()-begin; begin=time.monotonic()
        binding=dict(schema='CAPTURED_FOREST_COMPLETE_BINDING_V1',source=source,forest=digest(raw),
                     contract=identity(c),proof=identity(proof))
        with (bundle/'binding.json').open('xb') as s: s.write(compact(binding))
        phases['binding_publish_seconds']=time.monotonic()-begin; w.tick()
        receipt.update(status='COMPLETE',source=source,forest=digest(raw),binding=identity(binding),result=result,
            stats=dict(f.receipt['stats']),generation=generation,native=native,differential=comparison,
            environment=dict(python=sys.version,torch=torch.__version__,numpy=__import__('numpy').__version__,
                             scipy=__import__('scipy').__version__,torch_threads=torch.get_num_threads()))
    except BaseException as exc:
        receipt.update(status='REFUSED',exception=type(exc).__name__,message=str(exc)); raise
    finally:
        receipt.update(phases=phases,worker_before_receipt_seconds=time.monotonic()-start,
                       work=None if w is None else dict(w.counts),real_models=0,gpu_calls=0)
        save(root/'receipt.json',receipt)
    return receipt


def execute(root):
    from scripts.run_tensor_rows_controls import terminal_bytes
    begin=time.monotonic(); root.mkdir(parents=True,exist_ok=False); calls=[]; status='INCOMPLETE'; code={}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(name,argv,cwd=REPO):
        start=time.monotonic(); call=dict(name=name,hard_cap_seconds=300,argv=argv); calls.append(call)
        try:
            r=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,timeout=300)
            (root/(name+'.stdout')).write_bytes(r.stdout); (root/(name+'.stderr')).write_bytes(r.stderr)
            call.update(status='COMPLETE' if r.returncode==0 else 'ERROR',returncode=r.returncode)
            if r.returncode: raise RuntimeError(name+' failed; retained')
        except subprocess.TimeoutExpired as exc:
            call.update(status='TIMEOUT',returncode=None)
            (root/(name+'.stdout')).write_bytes(exc.stdout or b''); (root/(name+'.stderr')).write_bytes(exc.stderr or b''); raise
        finally:
            call['parent_observed_seconds']=time.monotonic()-start
            if call['parent_observed_seconds']>300: call['status']='TIMEOUT'
        if call['status']!='COMPLETE': raise TimeoutError('late result cannot be accepted')
    try:
        for name in FILES:
            p=root/'code'/name; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(REPO/name,p)
            code[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        save(root/'execution.json',dict(schema='FOREST_ENDPOINT_CONTROL_EXECUTION_V1',code=code,protocol=code[DESIGN],
            cases=list(CASES),sources=ANCHORS,python=sys.executable,archive_cap=LIMIT,work_cap=8000000,device='cpu',
            native={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'act').rglob('*.py'))}))
        save(root/'development_review.json',dict(initial_unit_failure='two unit runs assumed mixed must be UNKNOWN; it actually checked positive',
            repair='unchanged mixed source retained; explicit weak negative duals test UNKNOWN independently of source outcome',
            pre_archive_repairs=['prevent deletion of checked-handle seal','only check_proof issues certificate terminal',
                'bounded incremental proof bytes','exact decimal source margin','independent operator oracle',
                'mid-call deadline and exception preserve partial tile']))
        run('unit',[sys.executable,'-B','-m','unittest','scripts.test_forest_endpoints','scripts.test_source_dag',
                    'scripts.test_source_dag_audit','scripts.test_coefficient_dag','scripts.test_tensor_rows','-v'])
        for name in CASES:
            run(name,[sys.executable,'-B','-c',
                'import sys;from scripts.run_forest_endpoint_controls import worker;worker(sys.argv[1],sys.argv[2],float(sys.argv[3]))',
                str(root/name),name,str(time.monotonic()+299)])
        dest=root/'relocated'; dest.mkdir(); shutil.copytree(root/'code',dest/'code')
        names=['small40','mixed','guarded_w8','half']; anchors=[]
        for name in names:
            shutil.copytree(root/name/'bundle',dest/name)
            anchors.append(json.loads((root/name/'receipt.json').read_bytes())['binding'])
        run('relocated',[sys.executable,'-B','-I','-S','-c',
            'import sys,json,time;sys.path.insert(0,sys.argv[1]);from scoped_source.verify_forest_endpoints import verify;'
            'd=time.monotonic()+300;r=[verify(p,h,deadline=d) for p,h in zip(sys.argv[2::2],sys.argv[3::2])];'
            'assert not {"numpy","scipy","torch","highspy"}&set(sys.modules);print(json.dumps(r))',
            str(dest/'code'),*sum(([str(dest/n),h] for n,h in zip(names,anchors)),[])],cwd=dest)
        status='PASS_COMPONENT_CONTROLS'
    except BaseException as exc:
        status='FAILED_OR_REFUSED'; save(root/'failure.json',dict(exception=type(exc).__name__,message=str(exc))); raise
    finally:
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        summary=dict(status=status,calls=calls,code=code,archive_cap=LIMIT,payload_before_terminal_bytes=size,
                     outer_before_terminal_seconds=time.monotonic()-begin,real_models=0,gpu_calls=0)
        summary,raw=terminal_bytes(summary,size,LIMIT)
        with (root/'terminal.json').open('xb') as s: s.write(raw)
        if summary['status']=='ARCHIVE_LIMIT_REFUSED': raise ValueError('archive capacity refused')
    return summary


if __name__=='__main__':
    sys.path.insert(0,str(REPO)); p=argparse.ArgumentParser(); p.add_argument('root',type=Path)
    print(json.dumps(execute(p.parse_args().root.resolve()),indent=2))
