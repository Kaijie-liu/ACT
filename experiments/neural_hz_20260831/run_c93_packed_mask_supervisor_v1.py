"""Freeze complete inherited sources/tests and retain the bounded C93 outcome."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c93_packed_mask_20260913_v1'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((EXP/'results/c92_topology_first_20260913_v1/preregistered.json').read_text())
    end=json.loads((EXP/'results/c92_topology_first_20260913_v1/exit.json').read_text())
    hashes=dict(prior['source_sha256']);provenance=_provenance(ROOT)
    if (not end['all_declared_stages_passed'] or end['tests_count']!=2144 or end['tests_exit']!=0
        or end['source_drift'] or end['provenance_drift'] or provenance!=prior['provenance']
        or any(_sha256(EXP/n)!=s for n,s in hashes.items())):raise ValueError('unchanged complete C92 qualification required')
    names=['C93_PACKED_MASK_PREREG_20260913.md','c93_packed_mask_v1.py',
        'test_c93_packed_mask_v1.py','c93_packed_mask_worker_v1.py',Path(__file__).name,
        'C92_FUSED_MASK_PROGRAM_HANDOFF_20260913.md','CHECKPOINT_C92_TOPOLOGY_FIRST_20260913_SHA256SUMS',
        'results/c92_topology_first_20260913_v1/preregistered.json','results/c92_topology_first_20260913_v1/exit.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=prior['tests']+['test_c93_packed_mask_v1.py'];count=2186
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=count,cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,
        transient_cap_bytes=1024**3,entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        test_wall_cap_s=60,worker_wall_cap_s=240,inherited_runtime_suite_rerun=True,
        source_hash_authentication='separate_complete_C32_style_diagnostics_all_calls_reported',
        fresh_original_network_native_authorized=False,terminal_solve_or_promotion_authorized=False,
        historical_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,qualification_passed=False)
    try:
        testcmd=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider','-o','junit_family=legacy',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*testcmd,'--collect-only'],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        expected={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=count or len(set(ids))!=count
            or {s.split('::',1)[0] for s in ids}!=expected):raise ValueError('complete exact collection differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(count=len(ids),nodeids=ids,frozen_before_test_execution=True))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(testcmd,60)
        with (RUN/'tests.log').open('x') as stream:
            tested=subprocess.run([*testcmd,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tested.returncode or sorted(actual)!=sorted(ids)
            or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete qualification failed; target not started')
        record['qualification_passed']=True
        print(json.dumps(dict(event='complete_tests_passed',count=count,wall_s=record['test_wall_s'])),flush=True)
        record['worker_started']=True
        with (RUN/'worker.log').open('x') as stream:
            job=subprocess.run([sys.executable,str(EXP/'c93_packed_mask_worker_v1.py')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=job.returncode
        if job.returncode or not json.loads((RUN/'result.json').read_text())['completed']:
            raise ValueError('complete topology-first diagnostic rejected')
        record['all_declared_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=s for n,s in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record.get('all_declared_stages_passed') or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
