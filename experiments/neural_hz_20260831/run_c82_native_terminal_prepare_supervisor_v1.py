"""Freeze complete sources/tests; preserve every bounded terminal stage."""
import json
import os
from pathlib import Path
import pickle
import pickletools
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c82_native_terminal_20260913_v1'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((EXP/'results/c81_binned_inverse_20260913_v1/preregistered.json').read_text())
    end=json.loads((EXP/'results/c81_binned_inverse_20260913_v1/exit.json').read_text())
    hashes=dict(prior['source_sha256'])
    if (not end['all_declared_stages_passed'] or not end['qualification_passed'] or end['tests_count']!=1923 or end['tests_exit']!=0 or end['source_drift'] or end['provenance_drift']
            or any(_sha256(EXP/n)!=s for n,s in hashes.items())):raise ValueError('unchanged complete source/conversion qualification required')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('branch/production candidate drift')
    names=['C82_NATIVE_TERMINAL_PREREG_20260913.md','c82_native_terminal_v1.py',
        'test_c82_native_terminal_v1.py','c82_native_terminal_prepare_worker_v1.py',
        Path(__file__).name,'C81_BINNED_INVERSE_AUDIT_20260913.md',
        'results/c81_binned_inverse_20260913_v1/preregistered.json',
        'results/c81_binned_inverse_20260913_v1/exit.json',
        'results/c81_binned_inverse_20260913_v1/result.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=list(prior['tests'])
    if len(tests)!=86 or len(set(tests))!=86:raise ValueError('complete qualification files required')
    tests.append('test_c82_native_terminal_v1.py')
    required_count=1946
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=required_count,cpu_threads=1,gpu_enabled=False,
        address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,entries_cap=64_000_000,
        whole_work_cap_per_stage=256_000_000,test_wall_cap_s=60,
        preparation_wall_cap_s=60,soft_deadlines_used=False,compiler_receipt_sha256=_sha256(EXP/'C78_NATIVE_BUILD_20260913.json'),
        source_hash_authentication='separate_complete_C32_style_diagnostics_all_calls_reported',
        fresh_original_network_native_authorized=False,terminal_solve_or_promotion_authorized=False,
        historical_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,qualification_passed=False)
    try:
        testcmd=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*testcmd,'--collect-only'],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        expected={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=required_count or len(set(ids))!=required_count
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
        print(json.dumps(dict(event='complete_tests_passed',count=len(ids),wall_s=record['test_wall_s'])),flush=True)
        record['preparation_started']=True
        cmd=[sys.executable,str(EXP/'c82_native_terminal_prepare_worker_v1.py')]
        with (RUN/'preparation.log').open('x') as stream:
            job=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        record['preparation_exit']=job.returncode
        if job.returncode:raise ValueError('complete final affine preparation rejected')
        result=json.loads((RUN/'result.json').read_text())
        if not result['completed']:raise ValueError('complete final affine preparation incomplete')
        print(json.dumps(dict(event='complete_final_affine_preparation_passed',wall_s=result['wall_s'],work=result['diagnostic_work'])),flush=True)
        record['all_declared_stages_passed']=True
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record.get('all_declared_stages_passed') or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
