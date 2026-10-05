"""One frozen joint row-gauge screen; retain every outcome, no solver run."""
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
DIRECTORY=EXP/'results/c39_half_alias_row_gauge_20260911_v1'
PRIOR=EXP/'results/c38_all_consumer_coefficients_20260911_v1'
PRIOR_ANCHORS={
    'result.json':'d88f211576a719072d2b7ceffe8c6d46e9af737a925067a12624c4a1538d0ff9',
    'census.json':'05ac103b9fe9f8a6b3aaf06a8eee80896729190681e9bbd75b79adfdffd1b46a',
    'exit.json':'da5602676fbea3633ecb27645f25dbd664876133cb26bb3e3f22d1d4dc93ae84'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(SOURCE/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('C34 actual state drift')
    if any(_sha256(PRIOR/n)!=sha for n,sha in PRIOR_ANCHORS.items()):raise ValueError('completed C38 evidence drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('entire prior source freeze drift')
    names=[Path(__file__).name,'c39_half_alias_row_gauge_v1.py','c39_half_alias_census_v1.py','c39_half_alias_row_gauge_worker_v1.py',
        'test_c39_half_alias_row_gauge_v1.py','test_c39_half_alias_census_v1.py','C39_HALF_ALIAS_ROW_GAUGE_PREREG_20260911.md',
        'C38_ALL_CONSUMER_COEFFICIENT_AUDIT_20260911.md','C38_DYADIC_ROW_GAUGE_HANDOFF_20260911.md',
        'C38_COMPLETED_FAILURE_PROFILE_20260911.json',
        *(str(PRIOR/n) for n in PRIOR_ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c39_half_alias_row_gauge_v1.py','test_c39_half_alias_census_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c39_half_alias_row_gauge_worker_v1.py'),str(DIRECTORY)]
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
            old_archives_unchanged=all(_sha256(SOURCE/n)==sha for n,sha in ANCHORS.items())
                and all(_sha256(PRIOR/n)==sha for n,sha in PRIOR_ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
