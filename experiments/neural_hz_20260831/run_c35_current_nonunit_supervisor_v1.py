"""Freeze, test, run once and retain all outcomes; never rerun C34's solver."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
from experiments.neural_hz_20260831.run_c34_portable_restore_supervisor_v1 import ANCHORS,SOURCE
EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c35_current_nonunit_census_20260911_v1'
PORTABLE=EXP/'results/c34_portable_rhs_restore_20260911_v1'


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(SOURCE/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('old C34 archive drift')
    if _sha256(PORTABLE/'restore_guard.json')!='77a2eb7f4b35a705f746f695043b8ecd3e1edd0d6b54f4165ba74b2a065e31a9':
        raise ValueError('successful portable restore evidence drift')
    prior=json.loads((PORTABLE/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('old complete source freeze drift')
    names=[Path(__file__).name,'c35_current_uid_census_v1.py','c35_nonunit_exact_census_v1.py',
        'c35_current_nonunit_worker_v1.py','test_c35_current_nonunit_census_v1.py',
        'C35_CURRENT_NONUNIT_CENSUS_PREREG_20260911.md',str(PORTABLE/'restore_guard.json'),
        'C34_NONUNIT_STRUCTURE_HANDOFF_20260911.md','C14_EARLY_REJECTION_AUDIT_20260910.md']
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c35_current_nonunit_census_v1.py',*prior['tests']]
    provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c35_current_nonunit_worker_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,
        tests=tests,provenance=provenance,command=command,worker_wall_cap_s=240,test_wall_cap_s=60,
        address_space_bytes=16*1024**3,measured_transient_cap_bytes=1024**3,entry_cap=64_000_000,
        whole_diagnostic_work_cap=256_000_000,branch_work_cap=200_000_000,
        generator_native_or_solver_execution_authorized=False,old_archive_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        with (DIRECTORY/'tests.log').open('x') as f:
            result=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                *(str(EXP/n) for n in tests)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=result.returncode
        if result.returncode:return
        with (DIRECTORY/'worker.log').open('x') as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit_code']=result.returncode
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            old_archives_unchanged=all(_sha256(SOURCE/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
