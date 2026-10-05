"""One exclusive read-only source census, with automatic tests/log/exit retention."""
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
EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c57_shared_scalar_census_20260912_v2'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    hashes={}
    for record in ['results/c57_real_scalar_census_20260912_v1/preregistered.json',
                   'results/c56_inverse_live_20260912_v2/preregistered.json']:
        for name,sha in json.loads((EXP/record).read_text())['source_sha256'].items():
            if name in hashes and hashes[name]!=sha:raise ValueError('inherited source conflict')
            hashes[name]=sha
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=['C57_SHARED_SCALAR_V2_PREREG_20260912.md','results/c57_real_scalar_census_20260912_v1/result.json','results/c57_real_scalar_census_20260912_v1/exit.json','c57_scalar_consumer_probe_v2.py',
        'c57_real_scalar_census_v2.py','test_c57_real_scalar_census_v2.py','c57_real_scalar_worker_v2.py',
        Path(__file__).name,'CHECKPOINT_C56_GAUGED_CARRIER_20260912_SHA256SUMS',
        'C56_ACTUAL_SOURCE_HANDOFF_20260912.md','results/c53_logical_singletons_20260912_v2/result.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    tests=['test_c57_real_scalar_census_v2.py','test_c53_logical_singleton_census_v2.py']
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        stage='read_only_complete_real_source_general_scalar_diagnostic',focused_tests=tests,
        focused_total_cap_s=60,worker_cap_s=240,cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,
        transient_cap_bytes=1024**3,entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        unchanged_radix_auxiliary_cap=16384,unchanged_radix_added_entries_cap=131072,unchanged_radix_work_cap=16_000_000,
        complete_old_suite_or_generation_qualification_claimed=False,new_generator_native_or_solver_authorized=False,
        historical_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        commands=[sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in tests)]
        collected=subprocess.run([*commands,'--collect-only'],cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=60)
        with (DIRECTORY/'focused_collection.log').open('x') as stream:stream.write(collected.stdout)
        ids=[s for s in collected.stdout.splitlines() if s.startswith('experiments/neural_hz_20260831/') and '::' in s]
        if collected.returncode or len(ids)!=len(set(ids)) or len(ids)!=26:raise ValueError('focused collection failed')
        _atomic_exclusive_json(DIRECTORY/'focused_inventory.json',dict(nodeids=ids,count=len(ids)))
        left=60-(time.monotonic()-started)
        if left<=0:raise subprocess.TimeoutExpired(commands,60)
        with (DIRECTORY/'focused_tests.log').open('x') as stream:
            checked=subprocess.run([*commands,'--junitxml='+str(DIRECTORY/'focused_tests.xml')],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,timeout=left)
        record.update(focused_tests_exit=checked.returncode,focused_tests_count=len(ids),focused_total_wall_s=time.monotonic()-started)
        cases=ET.parse(DIRECTORY/'focused_tests.xml').findall('.//testcase')
        actual=[c.get('classname','').replace('.','/')+'.py::'+c.get('name','') for c in cases]
        if (checked.returncode or sorted(actual)!=sorted(ids)
                or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped'))):
            raise ValueError('focused exactness gate failed; no real source worker')
        with (DIRECTORY/'worker.log').open('x') as stream:
            worker=subprocess.run([sys.executable,str(EXP/'c57_real_scalar_worker_v2.py'),str(DIRECTORY)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=worker.returncode
        raise SystemExit(worker.returncode)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s']=exc.timeout;raise SystemExit(124)
    except Exception as exc:
        record['failure']=dict(type=type(exc).__name__,reason=str(exc));raise SystemExit(1)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in DIRECTORY.iterdir() if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
