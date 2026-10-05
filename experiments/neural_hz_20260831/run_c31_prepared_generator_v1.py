"""Freeze/test/run one original-source prepared generator with NEW full proof."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c31_prepared_generator_20260911_v1'
PRIOR=EXP/'results/c30_first_write_20260911_v1'
ANCHORS={
    'results/c30_first_write_20260911_v1/result.json':'7b0af27cfac7ae0826a740229d746287d626b97886d522a2bf47d4f289c5f5da',
    'results/c30_first_write_20260911_v1/exit.json':'33a4bd117a2279a7beea59036cd26661d330cbb1f006242db332c1a43ad6b560',
    'results/c27_owned_journal_20260911_v1/result.json':'5179eaf4753b0d81ff817fd6786845f09161303e66e91cbf4261dace638125fa',
    'results/c25_live_relu_20260911_v1/relu78.pickle':'685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba',
    'results/c5_first_terminal_20260905_v1/layer75.pickle':'d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed',
    'results/c9_live_relu_20260906_v1/relu78.pickle':'5bf82fc83205cd9b5f52187e164c70ce3c03abffd9d8bf352a643f38d2966a65',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('completed original input/reference drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('complete inherited source drift')
    names=[Path(__file__).name,'c31_prepared_owned_rows_v1.py','c31_prepared_emission_v1.py',
        'c31_prepared_report_audit_v1.py','test_c31_prepared_owned_v1.py','c31_prepared_generator_worker_v1.py',
        'C31_PREPARED_GENERATOR_PREREG_20260911.md','C30_NATIVE_INTEGRATION_HANDOFF_20260911.md',
        'CHECKPOINT_C30_FIRST_WRITE_20260911_SHA256SUMS',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c31_prepared_owned_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(EXP/'c31_prepared_generator_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,provenance=provenance,
        command=command,wall_cap_s=240,test_wall_cap_s=60,memory_gb=16,
        whole_work_cap=256_000_000,branch_work_cap=200_000_000,independent_report_work_cap=256_000_000,
        construction_cap_bytes=1024**3,entry_cap=64_000_000,new_original_expression_generator_authorized=True,
        new_complete_original_math_owner_UID_proof_authorized=True,new_native_or_solver_authorized=False,
        combined_native_or_full_LIVE_gate_claim_authorized=False,old_archive_writes_authorized=False,formal_gain=0))
    started=time.monotonic();record={'formal_gain':0}
    try:
        with (DIRECTORY/'tests.log').open('x') as stream:
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                *(str(EXP/n) for n in tests)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=r.returncode
        if r.returncode:return
        with (DIRECTORY/'worker.log').open('x') as stream:
            r=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit_code']=r.returncode
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
