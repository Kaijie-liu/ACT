"""Freeze/test/run one complete transplant diagnostic on the actual C25 source."""

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
DIRECTORY=EXP/'results/c26_transplant_census_20260911_v1'
PRIOR=EXP/'results/c25_live_relu_20260911_v1'
ANCHORS={'results/c25_live_relu_20260911_v1/relu78.pickle':'685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba',
    'results/c25_live_relu_20260911_v1/qualification.json':'88a754ad695a7da0ed2997a1a88542433ecceac9e5d6fa4932424e67a2deb886',
    'results/c25_live_relu_20260911_v1/exit.json':'e35d5460ceb2bfbb712ad1f10be1e7bf944511342790e6cc24cdf0d1d219b2ae',
    'results/c15_unit_row_splice_20260910_v1/spliced_hz.pickle':'6b93e5f929be286d762f9c0779c0b8d505440325e9b786f943df73ff0022dc30',
    'results/c15_unit_row_splice_20260910_v1/result.json':'1284c9a1c645387d1285f1be39967ed592597995f10c5aac74aed2a2d7e1f9cd',
    'results/c15_unit_row_splice_20260910_v1/exit.json':'002d9f5b1fac9999484639befa523e59d8e14085fc8627422e5e3a64ca32faff'}


def main():
    if DIRECTORY.exists(): raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()): raise ValueError('completed archive drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text())
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()): raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c26_tagged_transplant_v1.py','c26_transplant_audit_v1.py',
        'c26_transplant_census_worker_v1.py','test_c26_tagged_transplant_v1.py','C26_TRANSPLANT_PREREG_20260911.md',
        'C25_UNIT_INTEGRATION_HANDOFF_20260911.md','CHECKPOINT_C25_LIVE_20260911_SHA256SUMS',
        'results/c25_archive_restore_guard_20260911_v1/result.json',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c26_tagged_transplant_v1.py',*prior['tests']]
    provenance=_provenance(ROOT)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(EXP/'c26_transplant_census_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',{'source_sha256':hashes,'tests':tests,
        'provenance':provenance,'command':command,'wall_cap_s':240,'test_wall_cap_s':60,'memory_gb':16,
        'functional_metadata_work_cap':256_000_000,'independent_diagnostic_work_cap':256_000_000,
        'entry_cap':64_000_000,'construction_cap_bytes':1024**3,
        'new_fresh_HZ_generation_authorized':False,'new_native_consumer_authorized':False,
        'solver_authorized':False,'old_archive_writes_authorized':False,
        'only_provisional_metadata_compilation_and_complete_proof':True,'formal_gain':0})
    started=time.monotonic(); record={'formal_gain':0}
    try:
        with (DIRECTORY/'tests.log').open('x') as stream:
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                *(str(EXP/n) for n in tests)],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=r.returncode
        if r.returncode:return
        with (DIRECTORY/'worker.log').open('x') as stream:
            r=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        record['worker_exit_code']=r.returncode
    except subprocess.TimeoutExpired as exc: record['timeout_s']=exc.timeout
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record); print(json.dumps(record),flush=True)


if __name__=='__main__':main()
