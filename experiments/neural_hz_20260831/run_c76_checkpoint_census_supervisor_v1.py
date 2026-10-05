"""Exclusive diagnostic run; freeze and preserve all original C74 sources."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;RUN=EXP/'results/c76_checkpoint_census_20260913_v1'


def main():
    if RUN.exists():raise FileExistsError(RUN)
    prior=json.loads((EXP/'results/c74_native_binding_20260913_v2/preregistered.json').read_text())
    end=json.loads((EXP/'results/c74_native_binding_20260913_v2/exit.json').read_text())
    hashes=dict(prior['source_sha256'])
    if (not end['qualification_passed'] or end['tests_count']!=1841 or end['tests_exit']!=0
            or any(_sha256(EXP/n)!=s for n,s in hashes.items())):raise ValueError('unchanged qualified C74 source required')
    provenance=_provenance(ROOT)
    if provenance!=prior['provenance']:raise ValueError('original source/branch/provenance changed')
    names=['C76_CHECKPOINT_CENSUS_PREREG_20260913.md','c76_checkpoint_census_worker_v1.py',
        'run_c76_checkpoint_census_supervisor_v1.py','C75_RESTORE_INGESTION_AUDIT_20260913.md','C75_NATIVE_RESTORE_HANDOFF_20260913.md',
        'results/c75_restore_ingestion_20260913_v1/result.json',
        'results/c74_native_binding_20260913_v2/qualification.json',
        'results/c74_native_binding_20260913_v2/relu78.pickle',
        'results/c74_native_binding_20260913_v2/tests.xml','results/c74_native_binding_20260913_v2/exit.json']
    hashes.update({n:_sha256(EXP/n) for n in names})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if env.get('PYTHONOPTIMIZE') not in (None,'','0'):raise ValueError('normal assertions required')
    RUN.mkdir()
    _atomic_exclusive_json(RUN/'preregistered.json',dict(source_sha256=hashes,provenance=provenance,
        tests=prior['tests'],reused_unchanged_qualified_tests=1841,diagnostic_only=True,
        cpu_threads=1,gpu_enabled=False,hard_cap_s=60,soft_diagnostic_alarm_s=45,
        address_space_bytes=16*1024**3,transient_cap_bytes=1024**3,entries_cap=64_000_000,
        whole_work_cap=256_000_000,formal_gain=0))
    start=time.monotonic();record=dict(formal_gain=0,diagnostic_only=True)
    try:
        with (RUN/'worker.log').open('x') as log:
            job=subprocess.run([sys.executable,str(EXP/'c76_checkpoint_census_worker_v1.py')],cwd=ROOT,
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        record['worker_exit']=job.returncode
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-start,source_drift=any(_sha256(EXP/n)!=s for n,s in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={str(f.relative_to(RUN)):_sha256(f) for f in RUN.rglob('*') if f.is_file()})
        _atomic_exclusive_json(RUN/'exit.json',record);print(json.dumps(record),flush=True)
    if record.get('worker_exit')!=0 or record['source_drift'] or record['provenance_drift']:raise SystemExit(1)


if __name__=='__main__':main()
