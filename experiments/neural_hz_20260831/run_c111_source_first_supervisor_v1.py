# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full inherited qualification then complete ordinary source-first measurements."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c111_source_first_20260913_v1'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    previous=EXP/'results/c110_existing_factor_20260913_v1'
    prior=json.loads((previous/'preregistered.json').read_text())
    done=json.loads((previous/'exit.json').read_text())
    if (_sha256(previous/'exit.json')!='f52d736c95820c7076d631c68c24dc53fbbc9289d70efc8e38357716cd16ba19'
        or not done['all_stages_passed'] or done['tests_count']!=3224
        or done['source_drift'] or done['provenance_drift']):
        raise ValueError('complete C110 prerequisite missing')
    if any(_sha256(previous/n)!=h for n,h in done['artifacts'].items()):
        raise ValueError('C110 result/artifact drift')
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):
        raise ValueError('full inherited dependency drift')
    names=['C111_SOURCE_FIRST_PREREG_20260913.md','c111_form_reuse_v1.py',
        'c111_source_first_row_v1.py','c111_source_first_fixture_v1.py',
        'test_c111_source_first_v1.py','c111_source_first_worker_v1.py',
        'run_c111_source_first_supervisor_v1.py','C110_SOURCE_FIRST_HANDOFF_20260913.md']
    names += [str((previous/n).relative_to(EXP)) for n in ('preregistered.json','exit.json','tests.xml')]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=prior['tests']+['test_c111_source_first_v1.py']
    if len(tests)!=139 or len(set(tests))!=139:
        raise ValueError('complete inherited/new test files differ')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('production provenance drift')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('ordinary assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=3246,complete_test_wall_cap_s=60,stage_worker_wall_cap_s=240,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,
        whole_work_cap=256_000_000,entries_cap=64_000_000,shared_radix_caps=[16384,131072,16_000_000],
        all_comparison_numeric_arrays_retained=True,numeric_hash_traffic_in_token_pool=False,
        all_CPU_work_in_generation_cap=False,real_network_or_solver_run_authorized=False,
        source_runtime_LIVE_admitted=False,promotion_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(all_stages_passed=False,formal_gain=0)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                 *(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        files={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=3246 or len(set(ids))!=3246
            or {n.split('::',1)[0] for n in ids}!=files):
            raise ValueError('complete exact test node inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(nodeids=ids,count=len(ids)))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            tested=subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tested.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tested.returncode or sorted(actual)!=sorted(ids)
            or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete old/new mathematical qualification failed')
        print(json.dumps(dict(event='all_tests_passed',tests=len(ids),wall_s=record['test_wall_s'])),flush=True)
        if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('source drift after tests')
        with (RUN/'worker.log').open('x') as stream:
            worker=subprocess.run([sys.executable,str(EXP/'c111_source_first_worker_v1.py')],cwd=ROOT,
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=worker.returncode
        result=json.loads((RUN/'result.json').read_text())
        if worker.returncode or not result['completed'] or result['source_drift']:
            raise ValueError('complete source-first measurement failed')
        record.update(all_stages_passed=True,cases=len(result['cases']),
            numeric_winning_cases=result['numeric_winning_cases'],work=result['work'])
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['all_stages_passed'] or record['source_drift'] or record['provenance_drift']:
        raise SystemExit(1)


if __name__=='__main__':main()
