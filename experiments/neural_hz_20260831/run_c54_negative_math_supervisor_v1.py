"""Archive exact known v1 physical failures and one closed diagnostic measurement."""
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
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c54_exact_scalar_negative_20260912_v1'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    prior=json.loads((EXP/'results/c53_logical_singletons_20260912_v2/preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('old source drift')
    names=[Path(__file__).name,'c54_negative_math_worker_v1.py','c54_exact_dyadic_v1.py','c54_scalar_hz_v1.py',
        'c54_scalar_hz_audit_v1.py','c54_scalar_fixtures_v1.py','test_c54_scalar_hz_v1.py',
        'C54_EXACT_SCALAR_PREREG_20260912.md','C54_V1_NEGATIVE_MEASUREMENT_PREREG_20260912.md',
        'CHECKPOINT_C53_SOURCE_NORMALIZATION_20260912_SHA256SUMS','C53_GENERAL_SCALAR_HANDOFF_20260912.md']
    hashes.update({n:_sha256(EXP/n) for n in names});provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    expected=['test_complete_general_scalar_source_relation_and_physical_accounting[chain]','test_complete_general_scalar_source_relation_and_physical_accounting[conv_relu]']
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        expected_focused_test_failures=expected,focused_test_count=23,focused_tests_cap_s=60,worker_cap_s=240,
        fixed_cohorts=['chain','shared_add','conv_relu'],cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,
        transient_cap_bytes=1024**3,entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        diagnostic_only=True,representation_gate_relaxed=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,qualification_passed=False)
    try:
        with (DIRECTORY/'focused_tests.log').open('x') as f:
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider','--junitxml='+str(DIRECTORY/'focused_tests.xml'),
                str(EXP/'test_c54_scalar_hz_v1.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['focused_tests_exit']=r.returncode
        cases=ET.parse(DIRECTORY/'focused_tests.xml').findall('.//testcase')
        failures=[c.get('name') for c in cases if c.find('failure') is not None]
        if r.returncode!=1 or len(cases)!=23 or sorted(failures)!=sorted(expected) or any(c.find(n) is not None for c in cases for n in ('error','skipped')):
            raise ValueError('diagnostic cannot proceed after an unregistered test outcome')
        with (DIRECTORY/'worker.log').open('x') as f:
            r=subprocess.run([sys.executable,str(EXP/'c54_negative_math_worker_v1.py'),str(DIRECTORY)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=r.returncode
        raise SystemExit(r.returncode)
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in DIRECTORY.iterdir() if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
