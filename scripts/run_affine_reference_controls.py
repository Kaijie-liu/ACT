"""Frozen two-source V2 component archive; no dense run or output optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping

REPO=Path(__file__).resolve().parents[1]
PROTOCOL='docs/hz_affine_references_design_20261013.md'
LIMIT=16*2**20
ANCHORS={'mixed':'24155a8113346d6964b6a4497be0abc0e5dcebbb169c949931468d6fdbb92a22',
         'deep':'91a9cd3551e5f63e500e6db435a3039d6e158d7d671e88b7780a735267f4ba62'}
FILES=(PROTOCOL,'scoped_source/affine_references.py','scoped_source/check_affine_references.py',
 'scoped_source/affine_reference_controls.py','act/back_end/moe/affine_reference_operator.py',
 'scoped_source/source_dag.py','scoped_source/check_source_dag.py','scoped_source/source_dag_controls.py',
 'scoped_source/hz_request_store.py','scoped_source/hz_store_source.py','scoped_source/check_hz_store_source.py',
 'scoped_source/hz_store_sparse.py','scoped_source/hz_store_write.py','scoped_source/tensor_rows.py',
 'scoped_source/tensor_row_controls.py','scoped_source/factored_source.py','scoped_source/factored_io.py',
 'scoped_source/forest_endpoints.py','source_enclosure/format.py','upstream_source/checker.py',
 'scripts/test_affine_references.py','scripts/test_source_dag.py','scripts/test_forest_endpoints.py',
 'scripts/test_source_dag_audit.py','scripts/audit_source_dag_controls.py',
 'scripts/run_affine_reference_controls.py','scripts/audit_affine_reference_controls.py',
 'scripts/run_tensor_rows_controls.py')


def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)


def thaw(value):
    if isinstance(value,Mapping): return {k:thaw(v) for k,v in value.items()}
    if type(value) in (list,tuple): return [thaw(v) for v in value]
    return value


def worker(root,name,deadline):
    from scoped_source.hz_request_store import Work,digest,require
    from scoped_source.affine_references import build
    from scoped_source.check_affine_references import CheckedAffineForest
    from scoped_source.affine_reference_controls import deep,oracle
    from scoped_source.source_dag_controls import mixed
    started=time.monotonic(); w=Work(deadline); root=Path(root); phases={}; begin=started
    result=dict(status='INCOMPLETE',name=name,real_models=0,gpu_calls=0,full_request_certificates=0)
    try:
        require(name in ANCHORS,'frozen source name')
        source_root,source=(mixed if name=='mixed' else deep)(root/(name+'-fixture'))
        require(source==ANCHORS[name],'frozen source recipe identity')
        result.update(source=source,source_path=str(source_root.relative_to(root)))
        phases['fixture_seconds']=time.monotonic()-begin; begin=time.monotonic(); w.tick()
        raw,native=build(source_root,source,work=w)
        phases['build_and_encode_seconds']=time.monotonic()-begin; begin=time.monotonic()
        with (root/(name+'.forest.json')).open('xb') as f: f.write(raw)
        phases['forest_write_seconds']=time.monotonic()-begin; begin=time.monotonic(); w.tick()
        f=CheckedAffineForest(raw,source_root,source,expected=digest(raw),work=w)
        phases['independent_source_check_seconds']=time.monotonic()-begin; begin=time.monotonic()
        x,a,expected,trans,details=oracle(f,w)
        phases['exact_oracle_seconds']=time.monotonic()-begin; begin=time.monotonic()
        import torch
        torch.set_num_threads(1)
        from act.back_end.moe.affine_reference_operator import Operator
        n=Operator(f,w); val=lambda q:torch.tensor([[float(z) for z in row] for row in q],dtype=torch.float64,device='cpu')
        fx=n.forward(val(x)); ta=n.adjoint(val(a))
        torch.testing.assert_close(fx,val(expected),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(ta,val(trans),rtol=1e-12,atol=1e-12)
        phases['native_prepare_and_differential_seconds']=time.monotonic()-begin; begin=time.monotonic()
        save(root/(name+'.operator.json'),dict(source=source,forest=digest(raw),forward=fx.tolist(),adjoint=ta.tolist(),
            exact_forward=[[str(q) for q in row] for row in expected],exact_adjoint=[[str(q) for q in row] for row in trans],
            oracle=details,native=n.stats))
        phases['operator_serialization_seconds']=time.monotonic()-begin; w.tick()
        result.update(status='CHECKED_V2_SOURCE_AND_OPERATOR',forest_sha256=digest(raw),source_check=thaw(f.receipt),
            native_lowering=native,operator=n.stats,oracle=details,python=sys.version,torch_version=torch.__version__)
    except BaseException as exc:
        result.update(status='REFUSED',exception=type(exc).__name__,message=str(exc)); raise
    finally:
        result.update(phases=phases,work=dict(w.counts),worker_before_receipt_seconds=time.monotonic()-started)
        save(root/(name+'.receipt.json'),result)
    return result


def relocated(root,forest,source):
    from scoped_source.check_affine_references import check
    from scoped_source.hz_request_store import Work
    root=Path(root); start=time.monotonic()
    result=check((root/'forest.json').read_bytes(),root/'source',source,expected=forest,work=Work(start+300))
    if {'numpy','scipy','torch','highspy'}&set(sys.modules): raise ValueError('native import in portable checker')
    return result


def execute(root):
    from scripts.run_tensor_rows_controls import terminal_bytes
    start=time.monotonic(); root.mkdir(parents=True,exist_ok=False); calls=[]; status='INCOMPLETE'; code={}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(name,args,cwd=REPO):
        begin=time.monotonic(); entry=dict(name=name,hard_cap_seconds=300,argv=args); calls.append(entry)
        try:
            r=subprocess.run(args,cwd=cwd,env=env,capture_output=True,timeout=max(.001,300-(time.monotonic()-begin)))
            (root/(name+'.stdout')).write_bytes(r.stdout); (root/(name+'.stderr')).write_bytes(r.stderr)
            entry.update(status='COMPLETE' if r.returncode==0 else 'ERROR',returncode=r.returncode)
            if r.returncode: raise RuntimeError(name+' failed; no numerical retry')
        except subprocess.TimeoutExpired as e:
            entry.update(status='TIMEOUT',returncode=None)
            (root/(name+'.stdout')).write_bytes(e.stdout or b''); (root/(name+'.stderr')).write_bytes(e.stderr or b''); raise
        finally: entry['parent_observed_seconds']=time.monotonic()-begin
    try:
        for name in FILES:
            dest=root/'code'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(REPO/name,dest)
            code[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
        if dirty: raise ValueError('archive execution requires committed clean worktree')
        save(root/'execution.json',dict(protocol=code[PROTOCOL],code=code,git_head=head,python=sys.executable,source_anchors=ANCHORS,
            environment={k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            native_python_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((REPO/'act').rglob('*.py'))},archive_limit=LIMIT,
            dense='NOT_ADMITTED_STATIC_COMPLETE_QUERY_CAPACITY',dense_prefix_work=7202670,
            dense_native_minimum_extra=800622,dense_zero_endpoint_minimum_extra=1447977,
            dense_zero_with_prefix_minimum=8650647))
        save(root/'development_review.json',dict(first_unit_run='49/51 pass; two unregistered test expectations wrong',
            failures=['expected >2 reference levels; frozen deep has two levels and nonzero R squared',
                      'expected compensation in expert0/2; it occurs in router/2, router/4 and expert later layers'],
            repairs='test assertions only; no frozen source, tolerance, count or algorithm changed for these failures',
            added_before_archive=['zero weight and two nonzero center lifts, checked error-before-lift order',
                'unlowered suffix mutation','checker expansion forbidden','oracle arithmetic fully charged']))
        run('unit',[sys.executable,'-B','-m','unittest','scripts.test_affine_references','scripts.test_source_dag',
                    'scripts.test_forest_endpoints','scripts.test_source_dag_audit','-v'])
        for name in ('mixed','deep'):
            deadline=time.monotonic()+300
            run(name,[sys.executable,'-B','-c',
                'import sys;from scripts.run_affine_reference_controls import worker;worker(sys.argv[1],sys.argv[2],float(sys.argv[3]))',
                str(root),name,str(deadline)])
            receipt=json.loads((root/(name+'.receipt.json')).read_bytes()); dest=root/('relocated-'+name); dest.mkdir()
            shutil.copytree(root/'code',dest/'code'); shutil.copytree(root/receipt['source_path'],dest/'source')
            shutil.copyfile(root/(name+'.forest.json'),dest/'forest.json')
            run('relocated-'+name,[sys.executable,'-B','-I','-S','-c',
                'import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.run_affine_reference_controls import relocated;print(json.dumps(relocated(*sys.argv[2:])))',
                str(dest/'code'),str(dest),receipt['forest_sha256'],receipt['source']],cwd=dest)
        status='PASS_BOUNDED_V2_COMPONENT'
    except BaseException as exc:
        status='FAILED_OR_REFUSED'; save(root/'failure.json',dict(exception=type(exc).__name__,message=str(exc))); raise
    finally:
        payload=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        summary=dict(status=status,calls=calls,code=code,archive_cap=LIMIT,payload_before_terminal_bytes=payload,
            outer_before_terminal_seconds=time.monotonic()-start,real_models=0,gpu_calls=0,full_request_certificates=0)
        summary,raw=terminal_bytes(summary,payload,LIMIT)
        with (root/'terminal.json').open('xb') as f: f.write(raw)
        if summary['status']=='ARCHIVE_LIMIT_REFUSED': raise ValueError('archive capacity refused')
    return summary


if __name__=='__main__':
    sys.path.insert(0,str(REPO)); p=argparse.ArgumentParser(); p.add_argument('root',type=Path)
    print(json.dumps(execute(p.parse_args().root.resolve()),indent=2))
