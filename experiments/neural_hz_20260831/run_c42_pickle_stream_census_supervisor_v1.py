"""Frozen nonexecuting opcode census; no unpickler, new HZ or solver."""
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
DIRECTORY=EXP/'results/c42_pickle_stream_census_20260911_v1'
PRIOR=EXP/'results/c41_owned_checkpoint_gate_20260911_v1'
PRIOR_ANCHORS={
    'result.json':'a69033eff1f3e74e3f25392d2252a5fd3c7e4da7e41e8c6e1d9c3f76a999376c',
    'events.jsonl':'0e63f407451d1cbd95314955eb26d54f951f9c1082e8277d8e55df4e4a175b3a',
    'exit.json':'0bafce7157dd6f18b3e1fc363e2b932832aaf733c1154929fdee537c4e6ed7ac'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(SOURCE/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('C34 actual state drift')
    if any(_sha256(PRIOR/n)!=sha for n,sha in PRIOR_ANCHORS.items()):raise ValueError('closed C41 evidence drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('entire prior source freeze drift')
    names=[Path(__file__).name,'c42_pickle_stream_census_v1.py','c42_pickle_stream_census_worker_v1.py',
        'test_c42_pickle_stream_census_v1.py','C42_STREAM_CENSUS_PREREG_20260911.md',
        'C41_OWNED_CHECKPOINT_GATE_AUDIT_20260911.md','C41_DECODE_MEMORY_HANDOFF_20260911.md',
        'C41_CLOSED_DECODE_PROFILE_20260911.json',
        *(str(PRIOR/n) for n in PRIOR_ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c42_pickle_stream_census_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c42_pickle_stream_census_worker_v1.py'),str(DIRECTORY)]
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
