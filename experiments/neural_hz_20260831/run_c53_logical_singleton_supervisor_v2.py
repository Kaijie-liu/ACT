"""Exclusive diagnostic census, focused math tests; no full-suite substitute."""
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
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c53_logical_singletons_20260912_v2'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    prior=json.loads((EXP/'results/c53_source_normalization_20260912_v1/preregistered.json').read_text())
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source/archive drift')
    names=[Path(__file__).name,'c53_logical_singleton_census_v2.py','c53_logical_singleton_worker_v2.py',
        'test_c53_logical_singleton_census_v2.py','C53_LOGICAL_CENSUS_V2_PREREG_20260912.md',
        'C53_STRICT_ENVELOPE_THEOREM_20260912.md','CHECKPOINT_C52_SIGNED_FIRST_WRITE_20260911_SHA256SUMS',
        'C52_REAL_SOURCE_HANDOFF_20260911.md','C52_SIGNED_FIRST_WRITE_AUDIT_20260911.md',
        'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle']
    hashes.update({n:_sha256(EXP/n) for n in names});provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    tests=['test_c53_logical_singleton_census_v2.py','test_c53_source_normalization_v1.py','test_c52_signed_first_write_v1.py']
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        stage='read_only_complete_real_source_diagnostic',focused_tests=tests,focused_test_count=39,
        focused_test_cap_s=60,worker_cap_s=240,cpu_threads=1,gpu_enabled=False,address_space_bytes=16*1024**3,
        transient_cap_bytes=1024**3,entries_cap=64_000_000,whole_work_cap=256_000_000,branch_work_cap=200_000_000,
        complete_old_suite_or_generation_qualification_claimed=False,new_generator_native_or_solver_authorized=False,
        historical_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        with (DIRECTORY/'focused_tests.log').open('x') as f:
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                '--junitxml='+str(DIRECTORY/'focused_tests.xml'),*(str(EXP/n) for n in tests)],cwd=ROOT,env=env,
                stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['focused_tests_exit']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
        cases=ET.parse(DIRECTORY/'focused_tests.xml').findall('.//testcase')
        if len(cases)!=39 or any(case.find(n) is not None for case in cases for n in ('failure','error','skipped')):
            raise ValueError('all39 focused diagnostic/math tests must pass')
        with (DIRECTORY/'worker.log').open('x') as f:
            r=subprocess.run([sys.executable,str(EXP/'c53_logical_singleton_worker_v2.py'),str(DIRECTORY)],
                cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,artifacts={f.name:_sha256(f) for f in DIRECTORY.iterdir() if f.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
