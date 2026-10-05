"""Exclusive full inherited tests, complete bound, native proof and inverse."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c99_circuit_native_20260913_v2'
NAMES=['C99_SOURCE_PUBLICATION_PREREG_20260913.md','c99_unique_row_bound_v1.py',
    'c99_circuit_consumer_v2.py','c99_append_discovery_v2.py','c99_circuit_journal_v2.py',
    'c99_native_proof_v2.py','c99_writer_bound_v1.py','c99_native_worker_v2.py',
    'test_c99_circuit_native_v1.py','test_c99_unique_row_bound_v1.py','run_c99_native_supervisor_v2.py']


def main():
    if RUN.exists():raise FileExistsError(RUN)
    previous=EXP/'results/c99_circuit_native_20260913_v1'
    old=json.loads((previous/'preregistered.json').read_text());done=json.loads((previous/'exit.json').read_text())
    if (not done['all_declared_stages_passed'] or done['tests_count']!=2524
        or done['source_drift'] or done['provenance_drift']):raise ValueError('qualified complete C99v1 required')
    hashes=dict(old['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('frozen dependency drift')
    names=NAMES+['test_c99_source_publication_v2.py']+[str(p.relative_to(EXP)) for p in
        (previous/'preregistered.json',previous/'exit.json',previous/'preflight/result.json',previous/'component/result.json',previous/'restore/result.json') if p.exists()]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=old['tests']+['test_c99_source_publication_v2.py']
    if len(tests)!=114 or len(set(tests))!=114:raise ValueError('incomplete inherited test files')
    provenance=_provenance(ROOT)
    if provenance!=old['provenance']:raise ValueError('production branch/candidate drift')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    RUN.mkdir()
    for name in ('preflight','component','restore'):(RUN/name).mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=2527,complete_test_wall_cap_s=60,stage_worker_wall_cap_s=240,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,entries_cap=64_000_000,
        shared_radix_caps=[16384,131072,16_000_000],native_payload_separate_C32_boundary=True,
        stages=['all_inherited_and_new_tests','full_source_bound','full_native_component','complete_inverse_restore'],
        historical_writes_authorized=False,full_LIVE_or_solver_admission=False,formal_gain=0))
    started=time.monotonic();record=dict(all_declared_stages_passed=False,formal_gain=0)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        files={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=2527 or len(set(ids))!=2527
            or {n.split('::',1)[0] for n in ids}!=files):raise ValueError('exact inherited/new test inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(nodeids=ids,count=len(ids),frozen_before_test_execution=True))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            finished=subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=finished.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (finished.returncode or sorted(actual)!=sorted(ids)
            or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete original/new qualification failed')
        print(json.dumps(dict(event='all_tests_passed',tests=len(ids),wall_s=record['test_wall_s'])),flush=True)
        for stage in ('preflight','component','restore'):
            if stage=='component':
                _atomic_exclusive_json(RUN/stage/'input_binding.json',dict(preflight_result_sha256=_sha256(RUN/'preflight/result.json')))
            if stage=='restore':
                _atomic_exclusive_json(RUN/stage/'input_binding.json',dict(component_result_sha256=_sha256(RUN/'component/result.json')))
            record[stage+'_started']=True
            with (RUN/stage/'worker.log').open('x') as stream:
                worker=subprocess.run([sys.executable,str(EXP/'c99_native_worker_v2.py'),stage],cwd=ROOT,env=env,
                    stdout=stream,stderr=subprocess.STDOUT,timeout=240)
            record[stage+'_exit']=worker.returncode
            result=json.loads((RUN/stage/'result.json').read_text())
            print(json.dumps(dict(event='stage_exit',stage=stage,exit=worker.returncode,completed=result['completed'])),flush=True)
            if worker.returncode or not result['completed']:raise ValueError(stage+' failed; next stage not launched')
        record['all_declared_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['all_declared_stages_passed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
