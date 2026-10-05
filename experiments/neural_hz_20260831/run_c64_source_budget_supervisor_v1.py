"""Frozen C64 source-cost preflight with exact tests and automatic retention."""
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
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c64_source_budget_20260913_v1'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    previous=EXP/'results/c64_birth_generator_20260913_v1'
    old=json.loads((previous/'preregistered.json').read_text());exit_record=json.loads((previous/'exit.json').read_text())
    if (not exit_record['qualification_passed'] or exit_record['source_drift'] or exit_record['provenance_drift']
            or exit_record['tests_count']!=1466):raise ValueError('complete frozen generator qualification required')
    hashes=dict(old['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('qualified source drift')
    names=['C64_SOURCE_BUDGET_PREREG_20260913.md','c64_source_budget_v1.py','test_c64_source_budget_v1.py',
        'c64_source_budget_worker_v1.py',Path(__file__).name]
    names.extend(str(f.relative_to(EXP)) for f in previous.iterdir() if f.is_file())
    hashes.update({n:_sha256(EXP/n) for n in names});provenance=_provenance(ROOT)
    if provenance!=old['provenance']:raise ValueError('qualified candidate/provenance changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=['test_c64_source_budget_v1.py'],required_tests=4,reused_unchanged_qualification_tests=1466,
        reused_qualification_exit_sha256=_sha256(previous/'exit.json'),test_wall_cap_s=60,worker_cap_s=240,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,
        entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        actual_target_generator_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,actual_target_generator_started=False)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',str(EXP/'test_c64_source_budget_v1.py')]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (DIRECTORY/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith('experiments/') and '::' in s]
        if collection.returncode or len(ids)!=4 or len(set(ids))!=4:raise ValueError('exact source-cost test collection failed')
        _atomic_exclusive_json(DIRECTORY/'inventory.json',dict(nodeids=ids,count=4))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (DIRECTORY/'tests.log').open('x') as stream:
            test=subprocess.run([*command,'--junitxml='+str(DIRECTORY/'tests.xml')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=test.returncode,tests_count=4,test_wall_s=time.monotonic()-started)
        cases=ET.parse(DIRECTORY/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if test.returncode or sorted(actual)!=sorted(ids) or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped')):
            raise ValueError('source cost qualification failed; no actual-source worker')
        with (DIRECTORY/'worker.log').open('x') as stream:
            worker=subprocess.run([sys.executable,str(EXP/'c64_source_budget_worker_v1.py'),str(DIRECTORY)],cwd=ROOT,
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=worker.returncode;raise SystemExit(worker.returncode)
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout;raise SystemExit(124)
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc));raise SystemExit(1)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in DIRECTORY.iterdir() if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
