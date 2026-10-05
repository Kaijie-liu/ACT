"""Freeze/test/run one complete paired archived-row codec qualification."""

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
DIRECTORY=EXP/'results/c29_prepared_rows_20260911_v1'
PRIOR=EXP/'results/c28_consumer_discovery_20260911_v1'
ANCHORS={
    'results/c9_live_relu_20260906_v1/relu78.pickle':'5bf82fc83205cd9b5f52187e164c70ce3c03abffd9d8bf352a643f38d2966a65',
    'results/c25_live_relu_20260911_v1/relu78.pickle':'685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba',
    'results/c28_consumer_discovery_20260911_v1/plans.json':'1a0f0d72c2a2aebf8e4814116512e187a597e5160a27934168ea6543ed3b8b6c',
    'results/c28_consumer_discovery_20260911_v1/result.json':'c0d17f83f0d9d354a7452e0a99e5689afe8130337bca098c0eede7a8f42f9598',
    'results/c28_consumer_discovery_20260911_v1/exit.json':'77dffbda0b1303358dea110f745f2c5ed36a5800353ca459bc9d02a012a0a827',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('completed source/reference drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c29_prepared_row_v1.py','c29_prepared_row_worker_v1.py',
        'test_c29_prepared_row_v1.py','C29_PREPARED_ROW_PREREG_20260911.md',
        'C28_EMISSION_DISCOVERY_HANDOFF_20260911.md','CHECKPOINT_C28_DISCOVERY_20260911_SHA256SUMS',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c29_prepared_row_v1.py',*prior['tests']]
    provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(EXP/'c29_prepared_row_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',{'source_sha256':hashes,'tests':tests,'provenance':provenance,
        'command':command,'wall_cap_s':240,'test_wall_cap_s':60,'memory_gb':16,
        'combined_two_source_codec_work_cap':256_000_000,'independent_diagnostic_work_cap':256_000_000,
        'construction_cap_bytes':1024**3,'entry_cap':64_000_000,'fresh_original_generator_authorized':False,
        'native_consumer_or_solver_authorized':False,'old_archive_writes_authorized':False,
        'complete_generator_or_live_gate_claim_authorized':False,'formal_gain':0})
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
