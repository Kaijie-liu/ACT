"""Freeze/test/run one full actual first-write qualification, exclusive outputs."""

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
DIRECTORY=EXP/'results/c30_first_write_20260911_v1'
PRIOR=EXP/'results/c29_prepared_rows_20260911_v1'
ANCHORS={
    'results/c25_live_relu_20260911_v1/relu78.pickle':'685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba',
    'results/c26_transplant_census_20260911_v1/lineage.pickle':'55aa00df127ca394dc3e53d4b2923c8d6fc7bce543062b5cbf5057414e79b789',
    'results/c26_transplant_census_20260911_v1/result.json':'89f5b9d97500b947bdd970d0bb903c70b67a02ce35dee684692e91e00777f61f',
    'results/c26_transplant_census_20260911_v1/exit.json':'8b227b72baa3935e450095b5d711c41edb7532343b6988d7aea0a9f1dc556f75',
    'results/c15_unit_row_splice_20260910_v1/spliced_hz.pickle':'6b93e5f929be286d762f9c0779c0b8d505440325e9b786f943df73ff0022dc30',
    'results/c28_consumer_discovery_20260911_v1/plans.json':'1a0f0d72c2a2aebf8e4814116512e187a597e5160a27934168ea6543ed3b8b6c',
    'results/c28_consumer_discovery_20260911_v1/result.json':'c0d17f83f0d9d354a7452e0a99e5689afe8130337bca098c0eede7a8f42f9598',
    'results/c28_consumer_discovery_20260911_v1/exit.json':'77dffbda0b1303358dea110f745f2c5ed36a5800353ca459bc9d02a012a0a827',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('completed reference drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c30_first_write_v1.py','c30_append_discovery_v1.py','c30_first_write_worker_v1.py',
        'test_c30_first_write_v1.py','C30_FIRST_WRITE_PREREG_20260911.md',
        'C29_REAL_WRITER_HANDOFF_20260911.md','CHECKPOINT_C29_PREPARED_20260911_SHA256SUMS',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c30_first_write_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(EXP/'c30_first_write_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,provenance=provenance,
        command=command,wall_cap_s=240,test_wall_cap_s=60,memory_gb=16,
        writer_discovery_work_cap=256_000_000,independent_diagnostic_work_cap=256_000_000,
        construction_cap_bytes=1024**3,entry_cap=64_000_000,new_generator_or_native_solver_authorized=False,
        full_live_or_whole_work_gate_claim_authorized=False,old_archive_writes_authorized=False,formal_gain=0))
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
