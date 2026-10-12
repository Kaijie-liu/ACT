"""Frozen finite component archive. Not a production verifier supervisor."""
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
SOURCES=Path('/data1/Kane/MOE/runs/hz_tensor_rows_20261013_r1/sources')
ANCHORS={'small40':'28c41d0da9493f2b80de1dab910a769868f12240ac00dc55727b9a716dc8fe65',
         'dense':'0745027b58f00c608a9db921204cb2feae58273600fee658d4c9ccc3d8c19dbc'}
FILES=('scoped_source/source_dag.py','scoped_source/check_source_dag.py','scoped_source/source_dag_controls.py',
    'scoped_source/hz_request_store.py','scoped_source/hz_store_source.py','scoped_source/check_hz_store_source.py',
    'scoped_source/hz_store_sparse.py','scoped_source/hz_store_write.py','scoped_source/tensor_rows.py',
    'scoped_source/tensor_row_controls.py','scoped_source/factored_source.py','scoped_source/factored_io.py',
    'source_enclosure/format.py','upstream_source/checker.py','scripts/test_source_dag.py',
    'scripts/run_source_dag_controls.py','scripts/audit_source_dag_controls.py','scripts/run_tensor_rows_controls.py',
    'docs/hz_source_dag_design_20261013.md')
LIMIT=64*2**20


def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)


def worker(root,name,deadline):
    from scoped_source.hz_request_store import Work,digest
    from scoped_source.source_dag import build
    from scoped_source.check_source_dag import check
    started=time.monotonic(); work=Work(deadline); root=Path(root); cut=None if name=='small40' else ['expert0',2]
    phases={}; result=dict(status='INCOMPLETE',name=name,source=ANCHORS[name],cut=cut)
    try:
        raw,native=build(root/'sources'/name,ANCHORS[name],work=work,cut=cut); built=time.monotonic()
        phases['generation_and_encode_seconds']=built-started; phases.update(native)
        # Publish source representation even if subsequent checking refuses.
        with (root/(name+'.forest.json')).open('xb') as f: f.write(raw)
        written=time.monotonic(); phases['file_write_seconds']=written-built
        checked=check(raw,root/'sources'/name,ANCHORS[name],expected=digest(raw),work=work,cut=cut)
        phases['independent_parse_source_check_seconds']=time.monotonic()-written
        result.update(status=checked['status'],forest_sha256=digest(raw),check=checked)
    except BaseException as exc:
        result.update(status='REFUSED',exception=type(exc).__name__,message=str(exc))
        raise
    finally:
        result.update(phases=phases,work=dict(work.counts),worker_seconds=time.monotonic()-started,
                      real_models=0,gpu_calls=0,full_request_certificates=0)
        save(root/(name+'.receipt.json'),result)
    return result


def relocated(root,anchor,source):
    from scoped_source.check_source_dag import check
    from scoped_source.hz_request_store import Work
    root=Path(root); start=time.monotonic()
    result=check((root/'forest.json').read_bytes(),root/'source',source,
                 expected=anchor,work=Work(start+300))
    if {'numpy','scipy','torch','highspy'}&set(sys.modules): raise ValueError('native import in portable checker')
    return result


def execute(root):
    from scripts.run_tensor_rows_controls import terminal_bytes
    start=time.monotonic(); root.mkdir(parents=True,exist_ok=False); calls=[]; status='INCOMPLETE'
    code={}; sources={}; env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(name,argv,cwd=REPO,hard=300):
        begin=time.monotonic(); item=dict(name=name,hard_cap_seconds=hard,argv=argv); calls.append(item)
        try:
            r=subprocess.run(argv,cwd=cwd,capture_output=True,timeout=hard,env=env)
            (root/(name+'.stdout')).write_bytes(r.stdout); (root/(name+'.stderr')).write_bytes(r.stderr)
            item.update(status='COMPLETE' if r.returncode==0 else 'ERROR',returncode=r.returncode)
            if r.returncode: raise RuntimeError(name+' failed; retained')
        except subprocess.TimeoutExpired as e:
            item.update(status='TIMEOUT',returncode=None)
            (root/(name+'.stdout')).write_bytes(e.stdout or b''); (root/(name+'.stderr')).write_bytes(e.stderr or b''); raise
        finally: item['parent_observed_seconds']=time.monotonic()-begin
    try:
        for name in FILES:
            p=root/'code'/name; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(REPO/name,p)
            code[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        native={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'act').rglob('*.py'))}
        for name in ANCHORS:
            shutil.copytree(SOURCES/name,root/'sources'/name)
            sources[name]={str(p.relative_to(root/'sources'/name)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((root/'sources'/name).rglob('*')) if p.is_file()}
        save(root/'execution.json',dict(protocol=code['docs/hz_source_dag_design_20261013.md'],code=code,native=native,
            source_members=sources,source_anchors=ANCHORS,python=sys.executable,environment={k:env[k] for k in
                ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},archive_limit=LIMIT))
        save(root/'development_review.json',dict(initial_unit_result='12 pass, 1 test assertion failure',
            issue='mixed differential correctly visited 91 roots; unregistered test guessed >100',
            repair='assert exact 91 roots; no mathematical fixture or experimental threshold altered for this repair',
            pre_archive_repairs=['O(outputs) pending compensation tails, not duplicate dense bank',
                'ReLU expert scope comes from source graph, including expert-first ReLU',
                'prefix mode captured privately after identity binding'],
            mixed_guard_control='router x,-x,0; ties at zero and nonzero guard coefficients'))
        run('unit',[sys.executable,'-B','-m','unittest','scripts.test_source_dag','-v'])
        for name in ANCHORS:
            deadline=time.monotonic()+300
            run(name,[sys.executable,'-B','-c',
                'import sys;from scripts.run_source_dag_controls import worker;worker(sys.argv[1],sys.argv[2],float(sys.argv[3]))',
                str(root),name,str(deadline)])
        r=json.loads((root/'small40.receipt.json').read_bytes()); dest=root/'relocated'; dest.mkdir()
        shutil.copytree(root/'code',dest/'code'); shutil.copytree(root/'sources'/'small40',dest/'source')
        shutil.copyfile(root/'small40.forest.json',dest/'forest.json')
        run('relocated',[sys.executable,'-B','-I','-S','-c',
            'import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.run_source_dag_controls import relocated;print(json.dumps(relocated(*sys.argv[2:])))',
            str(dest/'code'),str(dest),r['forest_sha256'],ANCHORS['small40']],cwd=dest)
        status='PASS_COMPONENT_CONTROLS'
    except BaseException as exc:
        status='FAILED_OR_REFUSED'; save(root/'failure.json',dict(exception=type(exc).__name__,message=str(exc))); raise
    finally:
        payload=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        summary=dict(status=status,calls=calls,code=code,source_anchors=ANCHORS,archive_cap=LIMIT,
            payload_before_terminal_bytes=payload,outer_before_terminal_seconds=time.monotonic()-start,
            real_models=0,gpu_calls=0,full_request_certificates=0)
        summary,raw=terminal_bytes(summary,payload,LIMIT)
        with (root/'terminal.json').open('xb') as f: f.write(raw)
        if summary['status']=='ARCHIVE_LIMIT_REFUSED': raise ValueError('archive capacity refused')
    return summary


if __name__=='__main__':
    sys.path.insert(0,str(REPO)); p=argparse.ArgumentParser(); p.add_argument('root',type=Path)
    print(json.dumps(execute(p.parse_args().root.resolve()),indent=2))
