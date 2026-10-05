"""Freeze every inherited test/source and retain all fixed-run exit evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c47_bound_source_20260911_v1'
PRIOR=EXP/'results/c46_witness_composition_20260911_v1'
ANCHORS={'result.json':'dd3fe8a4592be04fa477e6fd42d769c193d256c4d453b5ae43ffe85639158426',
    'exit.json':'25d267305171b58b3bd87f038d16ddf6b40573dfcfd99355607b06ac4ed2ce83'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('prior C46 evidence drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c47_bound_half_source_v1.py','c47_source_fixture_v1.py','c47_source_payment_floor_v1.py',
        'test_c47_bound_half_source_v1.py','c47_bound_source_worker_v1.py',
        'C47_BOUND_SOURCE_PREREG_20260911.md','C47_BOUND_SOURCE_CONTRACT_20260911.md',
        'C46_WITNESS_COMPOSITION_AUDIT_20260911.md','C46_FRESH_BINDING_HANDOFF_20260911.md',
        'results/c34_changed_terminal_20260911_v1/terminal_gate.json',
        'results/c39_half_alias_row_gauge_20260911_v1/result.json',
        'results/c40_compact_half_gauge_20260911_v1/result.json',*(str(PRIOR/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c47_bound_half_source_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c47_bound_source_worker_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,
        tests=tests,provenance=provenance,command=command,worker_wall_cap_s=240,test_wall_cap_s=60,
        address_space_bytes=16*1024**3,measured_transient_cap_bytes=1024**3,entry_cap=64_000_000,
        whole_diagnostic_work_cap=256_000_000,branch_work_cap=200_000_000,
        actual_benchmark_source_native_or_solver_authorized=False,old_archive_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        with (DIRECTORY/'tests.log').open('x') as f:
            result=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                *(str(EXP/n) for n in tests)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=result.returncode
        if result.returncode:raise SystemExit(result.returncode)
        with (DIRECTORY/'worker.log').open('x') as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit_code']=result.returncode
        if result.returncode:raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s']=exc.timeout
        raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            old_archives_unchanged=all(_sha256(PRIOR/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
