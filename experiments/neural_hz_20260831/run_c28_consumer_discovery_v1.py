"""Freeze/test/execute one complete actual-source discovery qualification."""

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
DIRECTORY=EXP/'results/c28_consumer_discovery_20260911_v1'
PRIOR=EXP/'results/c27_owned_journal_20260911_v1'
ANCHORS={
    'results/c27_owned_journal_20260911_v1/result.json':'5179eaf4753b0d81ff817fd6786845f09161303e66e91cbf4261dace638125fa',
    'results/c27_owned_journal_20260911_v1/exit.json':'d64ed3602e28caa89b97ef0826f354ebbf6f7ad341f5585ed8732e6ded9e63ad',
    'results/c26_transplant_census_20260911_v1/lineage.pickle':'55aa00df127ca394dc3e53d4b2923c8d6fc7bce543062b5cbf5057414e79b789',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('completed reference drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c28_consumer_discovery_v1.py','c28_consumer_discovery_worker_v1.py',
        'test_c28_consumer_discovery_v1.py','C28_CONSUMER_DISCOVERY_PREREG_20260911.md',
        'C27_FUSED_DISCOVERY_HANDOFF_20260911.md','CHECKPOINT_C27_OWNED_JOURNAL_20260911_SHA256SUMS',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c28_consumer_discovery_v1.py',*prior['tests']]
    provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(EXP/'c28_consumer_discovery_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',{'source_sha256':hashes,'tests':tests,
        'provenance':provenance,'command':command,'wall_cap_s':240,'test_wall_cap_s':60,'memory_gb':16,
        'direct_discovery_work_cap':256_000_000,'independent_diagnostic_work_cap':256_000_000,
        'construction_cap_bytes':1024**3,'entry_cap':64_000_000,
        'fresh_generation_authorized':False,'native_consumer_or_solver_authorized':False,
        'old_archive_writes_authorized':False,'whole_live_or_integration_claim_authorized':False,'formal_gain':0})
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
