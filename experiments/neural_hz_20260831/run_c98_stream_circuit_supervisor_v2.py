"""One frozen qualification -> full bound -> actual physical HZ -> restore run."""
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
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c98_stream_circuit_20260913_v2'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    prior=json.loads((EXP/'results/c98_stream_circuit_20260913_v1/preregistered.json').read_text())
    hashes=dict(prior['source_sha256'])
    completed=json.loads((EXP/'results/c98_stream_circuit_20260913_v1/exit.json').read_text())
    if (not completed['all_declared_stages_passed'] or completed['tests_count']!=2486
            or completed['tests_exit']!=0 or completed['actual_worker_exit']!=0
            or completed['source_drift'] or completed['provenance_drift']):
        raise ValueError('complete terminal C98v1 execution required required')
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('frozen original/C66 source drift')
    names=["C98_PROOF_ROOTS_PREREG_20260913.md","c98_worker_v2.py","c98_restore_v2.py","test_c98_proof_roots_v2.py","run_c98_stream_circuit_supervisor_v2.py","results/c98_stream_circuit_20260913_v1/exit.json","results/c98_stream_circuit_20260913_v1/preflight/result.json","results/c98_stream_circuit_20260913_v1/actual/result.json"]
    names.append('results/c98_stream_circuit_20260913_v1/preregistered.json')
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=prior['tests']+['test_c98_proof_roots_v2.py']
    if len(tests)!=111 or len(set(tests))!=111:raise ValueError('complete old/new qualification file inventory differs')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('research branch/candidate changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    DIRECTORY.mkdir();(DIRECTORY/'actual').mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=2487,test_wall_cap_s=60,worker_cap_s=240,restore_cap_s=60,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,
        entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        original_radix_caps=[16384,131072,16_000_000],
        stages=['complete_tests','authenticated_unchanged_complete_source_bound_reuse','fresh_source_generator_full_proof_physical_gate','fresh_restore'],
        C31_separate_construction_and_offline_qualification_boundary=True,
        solver_native_default_or_full_LIVE_authorized=False,historical_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,qualification_passed=False,actual_generator_started=False)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (DIRECTORY/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        expected_files={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=2487 or len(set(ids))!=2487
                or {n.split('::',1)[0] for n in ids}!=expected_files):raise ValueError('complete exact qualification inventory differs')
        _atomic_exclusive_json(DIRECTORY/'inventory.json',dict(count=len(ids),nodeids=ids,frozen_before_test_execution=True))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (DIRECTORY/'tests.log').open('x') as stream:
            tests_result=subprocess.run([*command,'--junitxml='+str(DIRECTORY/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tests_result.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(DIRECTORY/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if tests_result.returncode or sorted(actual)!=sorted(ids) or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped')):
            raise ValueError('complete qualification failed; no actual-source work')
        record['qualification_passed']=True
        print(json.dumps(dict(event='complete_tests_passed',tests=record['tests_count'],wall_s=record['test_wall_s'])),flush=True)
        preflight_path=EXP/'results/c98_stream_circuit_20260913_v1/preflight/result.json'
        if _sha256(preflight_path)!='62367ea903f7244b0c7ed0aee0f49f5434f16c6cd8f05f60721b181b3e53ff31':
            raise ValueError('unchanged complete original aggregate preflight changed')
        before=json.loads(preflight_path.read_text())
        record['preflight_reused_from_v1']=True
        if not before['completed'] or not before['data']['bound']['work_caps_fit']:raise ValueError('missing full fitting bound')
        _atomic_exclusive_json(DIRECTORY/'actual/input_binding.json',dict(preregistered_sha256=_sha256(DIRECTORY/'preregistered.json'),
            complete_source_bound_sha256=_sha256(preflight_path)))
        record['actual_generator_started']=True
        print(json.dumps(dict(event='complete_fitting_source_bound',bound=before['data']['bound'])),flush=True)
        with (DIRECTORY/'actual/worker.log').open('x') as stream:
            actual_result=subprocess.run([sys.executable,str(EXP/'c98_worker_v2.py'),'actual'],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['actual_worker_exit']=actual_result.returncode
        if actual_result.returncode:raise ValueError('actual generator/source/physical gate rejected')
        _atomic_exclusive_json(DIRECTORY/'restore_input_binding.json',dict(actual_result_sha256=_sha256(DIRECTORY/'actual/result.json')))
        with (DIRECTORY/'restore.log').open('x') as stream:
            restored=subprocess.run([sys.executable,str(EXP/'c98_restore_v2.py')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        record['restore_exit']=restored.returncode
        if restored.returncode:raise ValueError('fresh physical archive restoration rejected')
        record['all_declared_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(DIRECTORY)):_sha256(f) for f in DIRECTORY.rglob('*') if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)
    if not record.get('all_declared_stages_passed') or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
