# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full frozen qualification, then one complete exact operator-cost batch."""
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
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c109_shared_partial_20260913_v1'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    previous=EXP/'results/c108_ordinary_phase_20260913_v2'
    prior=json.loads((previous/'preregistered.json').read_text())
    done=json.loads((previous/'exit.json').read_text())
    if (_sha256(previous/'exit.json')!='ac958288eff0b3a560d35c9add06373bd2c3efb2c4e108c634a4bc1f9451b134'
        or not done['fixture_passed'] or not done['diagnostic_boundary_observed']
        or done['source_drift'] or done['provenance_drift'] or done['timeout_s']!=240):
        raise ValueError('complete honestly closed C108 prerequisite missing')
    if any(_sha256(previous/n)!=h for n,h in done['artifacts'].items()):raise ValueError('C108 artifact drift')
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('full inherited dependency drift')
    names=['C109_SHARED_PARTIAL_PREREG_20260913.md','c109_shared_partial_v1.py',
        'test_c109_shared_partial_v1.py','c109_shared_partial_worker_v1.py',
        'run_c109_shared_partial_supervisor_v1.py','C108_EXACT_OPERATOR_HANDOFF_20260913.md']
    names += [str((previous/n).relative_to(EXP)) for n in ('preregistered.json','exit.json','fixture.xml')]
    hashes.update({n:_sha256(EXP/n) for n in names})
    original=json.loads((EXP/'results/c107_canonical_encoding_20260913_v1/preregistered.json').read_text())
    tests=original['tests']+['test_c108_ordinary_phase_v2.py','test_c109_shared_partial_v1.py']
    if len(tests)!=137 or len(set(tests))!=137:raise ValueError('full inherited/new inventory differs')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('production provenance drift')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('ordinary assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=tests,required_test_count=3204,complete_test_wall_cap_s=60,stage_worker_wall_cap_s=240,
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_bytes=1024**3,
        whole_work_cap=256_000_000,entries_cap=64_000_000,shared_radix_caps=[16384,131072,16_000_000],
        fixture_geometries=[[1,2],[3,4]],modes=['dense','scaled','centered','masked','shared'],
        extra_dense_grids=[[1,1],[1,2],[2,2],[2,3],[3,3]],
        all_comparison_numeric_arrays_retained=True,numeric_hash_traffic_in_token_pool=False,
        all_CPU_work_in_generation_cap=False,real_network_or_solver_run_authorized=False,
        source_runtime_LIVE_admitted=False,promotion_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(all_stages_passed=False,formal_gain=0)
    try:
        command=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collection=subprocess.run([*command,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=60)
        with (RUN/'collection.log').open('x') as stream:stream.write(collection.stdout)
        ids=[s for s in collection.stdout.splitlines() if s.startswith(('experiments/','act/')) and '::' in s]
        files={str((EXP/n).resolve().relative_to(ROOT)) for n in tests}
        if (collection.returncode or len(ids)!=3204 or len(set(ids))!=3204
            or {n.split('::',1)[0] for n in ids}!=files):raise ValueError('full exact test node inventory differs')
        _atomic_exclusive_json(RUN/'inventory.json',dict(nodeids=ids,count=len(ids)))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(command,60)
        with (RUN/'tests.log').open('x') as stream:
            tests_done=subprocess.run([*command,'--junitxml='+str(RUN/'tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(tests_exit=tests_done.returncode,tests_count=len(ids),test_wall_s=time.monotonic()-started)
        cases=ET.parse(RUN/'tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (tests_done.returncode or sorted(actual)!=sorted(ids)
            or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('complete old/new mathematical qualification failed')
        print(json.dumps(dict(event='all_tests_passed',tests=len(ids),wall_s=record['test_wall_s'])),flush=True)
        if any(_sha256(EXP/n)!=h for n,h in hashes.items()):raise ValueError('source drift after tests')
        with (RUN/'worker.log').open('x') as stream:
            worker=subprocess.run([sys.executable,str(EXP/'c109_shared_partial_worker_v1.py')],cwd=ROOT,
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=worker.returncode
        result=json.loads((RUN/'result.json').read_text())
        if worker.returncode or not result['completed'] or result['source_drift']:
            raise ValueError('complete prototype measurement failed')
        record.update(all_stages_passed=True,cases=len(result['cases']),
            numeric_winning_cases=result['numeric_winning_cases'],work=result['work'])
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=h for n,h in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if not record['all_stages_passed'] or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
