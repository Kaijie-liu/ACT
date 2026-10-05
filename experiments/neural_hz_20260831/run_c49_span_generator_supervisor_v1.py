"""Freeze and retain one complete C49 source generator qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c49_span_generator_20260911_v1'
PRIOR=EXP/'results/c48_alias_span_20260911_v1'
ANCHORS={'result.json':'e75f53f987487198f8420314773c8b3f769f9b3e2ff465965b35fc197754b6e5',
    'exit.json':'25a1aee21202467e7341432092ccd92f37beb915010fcf22c911c707d92e4ed6'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('C48 proof/census drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('complete inherited source drift')
    names=[Path(__file__).name,'c49_span_emission_v1.py','c49_span_owned_rows_v1.py',
        'c49_span_report_audit_v1.py','test_c49_span_generator_v1.py','c49_span_generator_worker_v1.py',
        'C49_SPAN_GENERATOR_PREREG_20260911.md','C49_SOURCE_CONSTRUCTION_CONTRACT_20260911.md',
        'C48_ALIAS_SPAN_AUDIT_20260911.md','C48_GENERATOR_HANDOFF_20260911.md',
        'results/c31_prepared_generator_20260911_v1/closed_hz.pickle',
        'results/c31_prepared_generator_20260911_v1/closed_proof.json',
        'results/c31_prepared_generator_20260911_v1/result.json',
        'results/c31_prepared_generator_20260911_v1/exit.json',*(str(PRIOR/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c49_span_generator_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c49_span_generator_worker_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,
        provenance=provenance,command=command,worker_wall_cap_s=240,test_wall_cap_s=60,address_space_bytes=16*1024**3,
        measured_transient_cap_bytes=1024**3,entry_cap=64_000_000,whole_generation_work_cap=256_000_000,
        nested_generation_branch_cap=200_000_000,offline_input_decode_work_cap=256_000_000,
        independent_report_work_cap=256_000_000,independent_report_branch_cap=200_000_000,
        unchanged_C31_full_source_proof_diagnostic_boundary=True,
        new_actual_generator_and_complete_source_proof_authorized=True,new_native_or_solver_authorized=False,
        original_archives_or_production_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        with (DIRECTORY/'tests.log').open('x') as f:
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                *(str(EXP/n) for n in tests)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
        with (DIRECTORY/'worker.log').open('x') as f:r=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit_code']=r.returncode
        if r.returncode:raise SystemExit(r.returncode)
    except subprocess.TimeoutExpired as exc:
        record['timeout_s']=exc.timeout;raise SystemExit(124)
    finally:
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            old_archives_unchanged=all(_sha256(PRIOR/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
