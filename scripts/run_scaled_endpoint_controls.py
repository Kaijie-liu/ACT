"""Six frozen finite CPU calls, hard supervisor, complete accounting and archive.

No dense source, CUDA, native solver or trained model is admitted here.
"""
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
DESIGN='docs/hz_scaled_batch_design_20261013.md'
LIMIT=16*2**20
CASES=('mixed','deep','clipped','zero','half','guarded')
ANCHORS={'mixed':'24155a8113346d6964b6a4497be0abc0e5dcebbb169c949931468d6fdbb92a22',
 'deep':'91a9cd3551e5f63e500e6db435a3039d6e158d7d671e88b7780a735267f4ba62',
 'clipped':'fec50a03b10e4d0f0c139b7702e3cdc0705e0c167eb7e5a08d35c31f14609806',
 'zero':'de658c8fa7bf0c7e090e0b31d7830d48c84822235b0fb84c70b9ff2fe0452a84',
 'half':'cf4dbc3dff90da9d71fc9a056e29cdefa69a2aa8dae768f08555a7b84fb4c8a0',
 'guarded':'0be644fc20d28ab6d811f4c933f58456d9039e7a1d513e38d41f929b92cfc8da'}
FILES=(DESIGN,'scoped_source/scaled_references.py','scoped_source/check_scaled_references.py',
 'scoped_source/scaled_endpoints.py','act/back_end/moe/scaled_forest_endpoints.py',
 'scoped_source/scaled_endpoint_controls.py','scoped_source/verify_scaled_endpoints.py',
 'scoped_source/source_dag.py','scoped_source/check_source_dag.py','scoped_source/source_dag_controls.py',
 'scoped_source/affine_references.py','scoped_source/check_affine_references.py','scoped_source/affine_reference_controls.py',
 'scoped_source/forest_endpoints.py','scoped_source/forest_endpoint_controls.py',
 'scoped_source/hz_request_store.py','scoped_source/hz_store_source.py','scoped_source/check_hz_store_source.py',
 'scoped_source/hz_store_sparse.py','scoped_source/hz_store_write.py','scoped_source/check_store_endpoints.py',
 'scoped_source/tensor_rows.py','scoped_source/tensor_row_controls.py','scoped_source/factored_source.py',
 'scoped_source/factored_io.py','scoped_source/coefficient_dag.py','scoped_source/coefficient_dag_controls.py',
 'source_enclosure/format.py','upstream_source/checker.py','scoped_proof/io.py',
 'scripts/test_scaled_endpoints.py','scripts/test_scaled_endpoint_archive.py','scripts/test_affine_references.py',
 'scripts/test_source_dag.py','scripts/test_forest_endpoints.py','scripts/test_source_dag_audit.py',
 'scripts/test_affine_reference_archive.py','scripts/audit_source_dag_controls.py',
 'scripts/run_source_dag_controls.py','scripts/run_tensor_rows_controls.py','scripts/run_affine_reference_controls.py',
 'scripts/audit_affine_reference_controls.py','scripts/run_store_device_controls.py','scripts/hz_store_supervised.py',
 'scripts/run_scaled_endpoint_controls.py','scripts/audit_scaled_endpoint_controls.py')


def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)


def call(root,name,argv,*,cwd,env,budget=300):
    """Parent time includes launching, waiting and log reception. Never accepts late success."""
    start=time.monotonic(); deadline=start+budget
    args=argv(deadline) if callable(argv) else argv
    record=dict(name=name,argv=args,hard_cap_seconds=budget,status='INCOMPLETE',returncode=None)
    try:
        r=subprocess.run(args,cwd=cwd,env=env,capture_output=True,timeout=max(.001,deadline-time.monotonic()))
        (root/(name+'.stdout')).write_bytes(r.stdout); (root/(name+'.stderr')).write_bytes(r.stderr)
        record.update(status='COMPLETE' if r.returncode==0 else 'ERROR',returncode=r.returncode)
    except subprocess.TimeoutExpired as exc:
        record['status']='TIMEOUT'
        (root/(name+'.stdout')).write_bytes(exc.stdout or b''); (root/(name+'.stderr')).write_bytes(exc.stderr or b'')
    except BaseException as exc:
        record.update(status='ERROR',exception=type(exc).__name__,message=str(exc))
    finally:
        record['parent_observed_seconds']=time.monotonic()-start
        if record['parent_observed_seconds']>=budget: record['status']='TIMEOUT'
    return record


def worker(root,name,deadline):
    started=time.monotonic(); root=Path(root); root.mkdir(exist_ok=False); w=None; phases={}
    receipt=dict(status='INCOMPLETE',name=name,real_models=0,gpu_calls=0)
    try:
        from scoped_source.hz_request_store import Work,compact,digest,identity,require
        w=Work(deadline)
        import torch
        torch.set_num_threads(1)
        from unittest.mock import patch
        from scoped_source.scaled_endpoint_controls import fixture,differential
        from scoped_source.scaled_references import build
        from scoped_source.check_scaled_references import CheckedScaledForest
        from scoped_source.scaled_endpoints import contract
        from act.back_end.moe.scaled_forest_endpoints import produce
        phases['imports_seconds']=time.monotonic()-started; begin=time.monotonic()
        require(name in CASES,'frozen finite case')
        source_root,source=fixture(root/'fixture',name); require(source==ANCHORS[name],'fixed captured source anchor')
        bundle=root/'bundle'; bundle.mkdir(); shutil.move(str(source_root),str(bundle/'source')); source_root=bundle/'source'
        phases['capture_and_place_seconds']=time.monotonic()-begin; begin=time.monotonic(); w.tick()
        raw,generation=build(source_root,source,work=w)
        phases['source_build_encode_seconds']=time.monotonic()-begin; begin=time.monotonic()
        with (bundle/'forest.json').open('xb') as stream: stream.write(raw)
        phases['forest_publish_seconds']=time.monotonic()-begin; begin=time.monotonic(); w.tick()
        f=CheckedScaledForest(raw,source_root,source,expected=digest(raw),work=w)
        phases['source_check_seconds']=time.monotonic()-begin; begin=time.monotonic()
        source_work=dict(w.counts); c=contract(f); partial=root/'partial'; partial.mkdir()
        def event(i,raw):
            with (partial/f'{i:05d}.json').open('xb') as stream: stream.write(raw)
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('physical CUDA forbidden')):
            proof,result,native=produce(f,c,work=w,width=8,event=event)
        phases['complete_candidate_and_check_seconds']=time.monotonic()-begin; begin=time.monotonic()
        for filename,value in (('contract.json',c),('proof.json',proof)):
            with (bundle/filename).open('xb') as stream: stream.write(compact(value))
        binding=dict(schema='CAPTURED_SCALED_COMPLETE_BINDING_V3',source=source,forest=digest(raw),
                     contract=identity(c),proof=identity(proof))
        with (bundle/'binding.json').open('xb') as stream: stream.write(compact(binding))
        phases['proof_and_binding_publish_seconds']=time.monotonic()-begin; production_work=dict(w.counts)
        begin=time.monotonic(); comparison=differential(f,c,proof,w)
        require(comparison['result']==result,'expanded checker result equality')
        phases['test_only_expanded_differential_seconds']=time.monotonic()-begin; w.tick()
        receipt.update(status='COMPLETE_FINITE_SCALED_CONTROL',source=source,forest=digest(raw),binding=identity(binding),
            result=result,stats=dict(f.receipt['stats']),generation=generation,native=native,differential=comparison,
            source_work=source_work,production_work=production_work,
            environment=dict(python=sys.version,torch=torch.__version__,torch_threads=torch.get_num_threads()))
    except BaseException as exc:
        receipt.update(status='REFUSED',exception=type(exc).__name__,message=str(exc)); raise
    finally:
        receipt.update(phases=phases,worker_before_receipt_seconds=time.monotonic()-started,
                       work=None if w is None else dict(w.counts))
        save(root/'receipt.json',receipt)
    return receipt


def execute(root):
    from scripts.run_tensor_rows_controls import terminal_bytes
    begin=time.monotonic(); root=Path(root); root.mkdir(parents=True,exist_ok=False); calls=[]; code={}; status='INCOMPLETE'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(name,args,cwd=REPO):
        result=call(root,name,args,cwd=cwd,env=env); calls.append(result)
        if result['status']!='COMPLETE': raise RuntimeError(name+' failed; retained, no numerical retry')
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):
            raise ValueError('archive execution requires committed clean worktree')
        for name in FILES:
            path=root/'code'/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(REPO/name,path)
            code[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        save(root/'execution.json',dict(schema='SCALED_ENDPOINT_CONTROL_EXECUTION_V3',code=code,
            protocol=code[DESIGN],git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            source_anchors=ANCHORS,cases=list(CASES),width=8,python=sys.executable,archive_cap=LIMIT,work_cap=8000000,
            dense='NOT_ADMITTED',gpu='NOT_POLLED_OR_RUN',
            native_python_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((REPO/'act').rglob('*.py'))},
            environment={k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
        save(root/'development_review.json',dict(initial_failure='test import SyntaxError from unparenthesized multiline with',
            repair='test syntax only; no numerical recipe/threshold change',
            source_review='first-affine checker retains only compressed <=2-term tails, not per-layer dense dict copies',
            guarantee='finite source correctness; no trained model, CUDA timing or external result'))
        run('unit',[sys.executable,'-B','-m','unittest','scripts.test_scaled_endpoints','scripts.test_scaled_endpoint_archive',
            'scripts.test_affine_references','scripts.test_source_dag','scripts.test_forest_endpoints',
            'scripts.test_source_dag_audit','scripts.test_affine_reference_archive','-v'])
        for name in CASES:
            run(name,lambda deadline,n=name:[sys.executable,'-B','-c',
                'import sys;from scripts.run_scaled_endpoint_controls import worker;worker(sys.argv[1],sys.argv[2],float(sys.argv[3]))',
                str(root/n),n,str(deadline)])
        dest=root/'relocated'; dest.mkdir(); shutil.copytree(root/'code',dest/'code'); anchors=[]
        for name in CASES:
            shutil.copytree(root/name/'bundle',dest/name)
            anchors.append(json.loads((root/name/'receipt.json').read_bytes())['binding'])
        run('relocated',[sys.executable,'-B','-I','-S','-c',
            'import sys,json,time;sys.path.insert(0,sys.argv[1]);from scoped_source.verify_scaled_endpoints import verify;'
            'deadline=time.monotonic()+300;result=[verify(p,h,deadline=deadline) for p,h in zip(sys.argv[2::2],sys.argv[3::2])];'
            'assert not {"torch","numpy","scipy","highspy"}&set(sys.modules);print(json.dumps(result))',
            str(dest/'code'),*sum(([str(dest/n),h] for n,h in zip(CASES,anchors)),[])],cwd=dest)
        status='PASS_FINITE_SCALED_ENDPOINT_CONTROLS'
    except BaseException as exc:
        status='FAILED_OR_REFUSED'; save(root/'failure.json',dict(exception=type(exc).__name__,message=str(exc))); raise
    finally:
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        summary=dict(status=status,calls=calls,code=code,archive_cap=LIMIT,payload_before_terminal_bytes=size,
                     outer_before_terminal_seconds=time.monotonic()-begin,real_models=0,gpu_calls=0,dense='NOT_ADMITTED')
        summary,raw=terminal_bytes(summary,size,LIMIT)
        with (root/'terminal.json').open('xb') as stream: stream.write(raw)
        if summary['status']=='ARCHIVE_LIMIT_REFUSED': raise ValueError('archive capacity refused')
    return summary


if __name__=='__main__':
    sys.path.insert(0,str(REPO)); parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    print(json.dumps(execute(parser.parse_args().root.resolve()),indent=2))
