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
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c66_packed_frontier_20260913_v1'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    prior=json.loads((EXP/'results/c65_owned_normal_20260913_v1/preregistered.json').read_text())
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('frozen original/C64 source drift')
    names=['C66_PACKED_FRONTIER_PREREG_20260913.md','c66_packed_frontier_v1.py','c66_birth_emission_v1.py',
        'c66_source_budget_v1.py','c66_source_budget_worker_v1.py','c66_actual_generator_worker_v1.py',
        'c66_restore_worker_v1.py','test_c66_birth_emission_v1.py','test_c66_source_budget_v1.py',
        'test_c66_full_source_v1.py','test_c66_packed_frontier_v1.py',Path(__file__).name,
        'C65_OWNED_NORMAL_AUDIT_20260913.md','C65_PACKED_FRONTIER_HANDOFF_20260913.md',
        'C65_ARCHIVE_INTEGRITY_20260913.json','CHECKPOINT_C65_OWNED_NORMAL_20260913_SHA256SUMS',
        'results/c65_owned_normal_20260913_v1/exit.json','results/c65_owned_normal_20260913_v1/actual/result.json',
        'results/c65_owned_normal_20260913_v1/actual/events.jsonl','results/c65_owned_normal_20260913_v1/preflight/result.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=[*prior['tests'],'test_c66_birth_emission_v1.py','test_c66_source_budget_v1.py',
        'test_c66_full_source_v1.py','test_c66_packed_frontier_v1.py']
    if len(tests)!=66 or len(set(tests))!=66:raise ValueError('complete old/new qualification file inventory differs')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('research branch/candidate changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    DIRECTORY.mkdir();(DIRECTORY/'preflight').mkdir();(DIRECTORY/'actual').mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=1540,test_wall_cap_s=60,worker_cap_s=240,restore_cap_s=60,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,
        entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        original_radix_caps=[16384,131072,16_000_000],
        stages=['complete_tests','complete_source_bound','fresh_source_generator_full_proof_physical_gate','fresh_restore'],
        C31_separate_construction_and_offline_qualification_boundary=True,
        solver_native_default_or_full_LIVE_authorized=False,historical_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,qualification_passed=False,actual_generator_started=False)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (DIRECTORY/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        expected_files={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=1540 or len(set(ids))!=1540
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
        with (DIRECTORY/'preflight/worker.log').open('x') as stream:
            preflight=subprocess.run([sys.executable,str(EXP/'c66_source_budget_worker_v1.py'),str(DIRECTORY/'preflight')],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['preflight_exit']=preflight.returncode
        if preflight.returncode:raise ValueError('complete source-bound gate rejected; no generator')
        before=json.loads((DIRECTORY/'preflight/result.json').read_text())
        if not before['completed'] or not before['data']['bound']['work_caps_fit']:raise ValueError('missing full fitting bound')
        _atomic_exclusive_json(DIRECTORY/'actual/input_binding.json',dict(preregistered_sha256=_sha256(DIRECTORY/'preregistered.json'),
            complete_source_bound_sha256=_sha256(DIRECTORY/'preflight/result.json')))
        record['actual_generator_started']=True
        with (DIRECTORY/'actual/worker.log').open('x') as stream:
            actual_result=subprocess.run([sys.executable,str(EXP/'c66_actual_generator_worker_v1.py'),str(DIRECTORY/'actual')],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['actual_worker_exit']=actual_result.returncode
        if actual_result.returncode:raise ValueError('actual generator/source/physical gate rejected')
        _atomic_exclusive_json(DIRECTORY/'restore_input_binding.json',dict(actual_result_sha256=_sha256(DIRECTORY/'actual/result.json')))
        with (DIRECTORY/'restore.log').open('x') as stream:
            restored=subprocess.run([sys.executable,str(EXP/'c66_restore_worker_v1.py')],cwd=ROOT,env=env,
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
