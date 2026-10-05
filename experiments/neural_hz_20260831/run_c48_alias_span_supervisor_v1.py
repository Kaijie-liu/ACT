"""One exclusive full-source routing census; retain failures and drift checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c48_alias_span_20260911_v1'
PRIOR=EXP/'results/c47_bound_source_20260911_v1'
ANCHORS={'result.json':'787ac99eb2fffbeefebadbe27fd8846debcd8d4129ffe8233d1a02fc2aaf5e43',
    'exit.json':'c2874a13e0090729e9a1d9b0fab66ccc31c96f15ab920e4f386f1c16b1fb16a2'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('C47 evidence drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c48_empty_alias_span_v1.py','c48_alias_span_census_v1.py',
        'test_c48_empty_alias_span_v1.py','c48_alias_span_worker_v1.py',
        'C48_ALIAS_SPAN_PREREG_20260911.md','C48_ALIAS_SPAN_THEOREM_20260911.md',
        'C47_BOUND_SOURCE_AUDIT_20260911.md','C47_FUSED_SOURCE_HANDOFF_20260911.md',
        'results/c31_prepared_generator_20260911_v1/result.json',
        'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle',
        'results/c9_checkpoint_qualification_20260905_v1/result.json',
        'C9_INTEGRATED_V1_CLOSED_CHECKPOINT_PREREG_20260905.md',*(str(PRIOR/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c48_empty_alias_span_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c48_alias_span_worker_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,
        provenance=provenance,command=command,worker_wall_cap_s=240,test_wall_cap_s=60,
        address_space_bytes=16*1024**3,measured_transient_cap_bytes=1024**3,entry_cap=64_000_000,
        whole_diagnostic_work_cap=256_000_000,nested_source_branch_cap=200_000_000,
        only_pre_registered_original_source_read_authorized=True,new_generator_native_or_solver_authorized=False,
        old_archive_writes_authorized=False,formal_gain=0))
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
        record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            old_archives_unchanged=all(_sha256(PRIOR/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
