"""Freeze once, save focused tests and bounded prototype result automatically."""
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
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c55_binary64_lift_20260912_v1'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    prior=json.loads((EXP/'results/c54_packed_exact_scalar_20260912_v2/preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('old source drift')
    names=['C55_BINARY64_LIFT_PREREG_20260912.md','c55_binary64_lift_v1.py','c55_binary64_lift_audit_v1.py',
        'test_c55_binary64_lift_v1.py','c55_binary64_math_worker_v1.py',Path(__file__).name,
        'CHECKPOINT_C54_EXACT_SCALAR_20260912_SHA256SUMS','C54_NATIVE_REALIZATION_HANDOFF_20260912.md']
    hashes.update({n:_sha256(EXP/n) for n in names});provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        focused_test_count=22,focused_tests_cap_s=60,worker_cap_s=240,cohorts=['chain','shared_add','conv_relu'],
        cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,
        entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        radix_auxiliary_cap=16384,radix_added_entries_cap=131072,radix_work_cap=16_000_000,
        mathematical_prototype_only=True,full_suite_qualification=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,full_suite_qualification_passed=False)
    try:
        with (DIRECTORY/'focused_tests.log').open('x') as stream:
            tests=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider','--junitxml='+str(DIRECTORY/'focused_tests.xml'),
                str(EXP/'test_c55_binary64_lift_v1.py')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        record['focused_tests_exit']=tests.returncode;cases=ET.parse(DIRECTORY/'focused_tests.xml').findall('.//testcase')
        if tests.returncode!=0 or len(cases)!=22 or any(c.find(n) is not None for c in cases for n in ('failure','error','skipped')):
            raise ValueError('focused mathematical gate rejected; no measured worker')
        with (DIRECTORY/'worker.log').open('x') as stream:
            result=subprocess.run([sys.executable,str(EXP/'c55_binary64_math_worker_v1.py'),str(DIRECTORY)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=result.returncode;raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in DIRECTORY.iterdir() if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
