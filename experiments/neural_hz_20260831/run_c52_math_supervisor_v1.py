"""Focused mathematical prototype gate only, never complete target admission."""
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
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c52_signed_first_write_math_20260911_v1'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    prior=json.loads((EXP/'results/c51_source_custody_20260911_v1/preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source/history changed')
    files=[Path(__file__).name,'c52_math_worker_v1.py','c52_signed_first_write_v1.py','c52_signed_first_write_audit_v1.py',
        'c52_neural_source_fixtures_v1.py','test_c52_signed_first_write_v1.py','C52_SIGNED_FIRST_WRITE_PREREG_20260911.md',
        'C52_SIGNED_FIRST_WRITE_CONTRACT_20260911.md','C51_SOURCE_CUSTODY_AUDIT_20260911.md','C51_COMMON_STRUCTURE_HANDOFF_20260911.md']
    hashes.update({n:_sha256(EXP/n) for n in files});provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions are required')
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        stage='math_equivalence_prototype_only',fixed_cohorts=['chain','shared_add','conv_relu'],copies_per_cohort=128,
        focused_tests=['test_c52_signed_first_write_v1.py'],focused_test_count=16,focused_test_cap_s=60,math_worker_cap_s=240,
        normal_pytest_no_assertion_compiler_hooks=True,original_full_suite_or_actual_target_authorized=False,
        native_solver_or_family_expansion_authorized=False,source_input_and_both_states_inside_one_measured_transaction=True,
        cpu_threads=1,gpu_enabled=False,address_space_cap_bytes=16*1024**3,transient_cap_bytes=1024**3,
        entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0,actual_target_authorized=False)
    try:
        with (DIRECTORY/'focused_tests.log').open('x') as f:
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                '--junitxml='+str(DIRECTORY/'focused_tests.xml'),str(EXP/'test_c52_signed_first_write_v1.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['focused_tests_exit']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
        cases=ET.parse(DIRECTORY/'focused_tests.xml').findall('.//testcase')
        if len(cases)!=16 or any(case.find(n) is not None for case in cases for n in ('failure','error','skipped')):
            raise ValueError('all sixteen focused mathematical tests must pass')
        with (DIRECTORY/'math_worker.log').open('x') as f:
            r=subprocess.run([sys.executable,str(EXP/'c52_math_worker_v1.py'),str(DIRECTORY)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['math_worker_exit']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in DIRECTORY.iterdir() if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
