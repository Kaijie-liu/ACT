"""Freeze all inherited evidence and unchanged consumers; synthetic run only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent
DIRECTORY=EXP/'results/c44_id_consumers_20260911_v1'
PRIOR=EXP/'results/c43_paged_stream_census_20260911_v1'
PRIOR_ANCHORS={
    'result.json':'6abbb950e41e7531c2683bf9e5afa4d7d28fe3c4bd6ac025ea6b5136b947371a',
    'census.json':'e024facd1d81d3efbbbf7afd8a65e8f1d067889462bf1ddff8b46b6f7f548941',
    'exit.json':'0185d03c3ae2a6e9a2df7b61ff68dcde44b91e54cfeb3de5fe4428e40c99511f'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in PRIOR_ANCHORS.items()):raise ValueError('closed C43 evidence drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('entire prior source freeze drift')
    names=[Path(__file__).name,'c44_var_ids_v1.py','c44_id_consumers_v1.py','c44_id_consumers_worker_v1.py',
        'test_c44_var_ids_v1.py','test_c44_id_consumers_v1.py','C44_ID_CONSUMERS_PREREG_20260911.md',
        'C43_METADATA_CONSTRUCTION_HANDOFF_20260911.md',*(str(PRIOR/n) for n in PRIOR_ANCHORS),
        *(str(ROOT/n) for n in ('act/back_end/core.py','act/back_end/layer_util.py','act/back_end/layer_schema.py',
            'act/back_end/hybridz_tf/hybridz_tf.py','act/pipeline/verification/batchnorm_graph.py'))]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c44_var_ids_v1.py','test_c44_id_consumers_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c44_id_consumers_worker_v1.py'),str(DIRECTORY)]
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
            old_archives_unchanged=all(_sha256(PRIOR/n)==sha for n,sha in PRIOR_ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
