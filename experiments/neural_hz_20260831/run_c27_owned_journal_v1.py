"""Freeze and run one fresh provisional ownership-journal diagnostic."""

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
DIRECTORY=EXP/'results/c27_owned_journal_20260911_v1'
PRIOR=EXP/'results/c26_transplant_census_20260911_v1'
ANCHORS={
    'results/c26_transplant_census_20260911_v1/lineage.pickle':'55aa00df127ca394dc3e53d4b2923c8d6fc7bce543062b5cbf5057414e79b789',
    'results/c26_transplant_census_20260911_v1/result.json':'89f5b9d97500b947bdd970d0bb903c70b67a02ce35dee684692e91e00777f61f',
    'results/c26_transplant_census_20260911_v1/exit.json':'8b227b72baa3935e450095b5d711c41edb7532343b6988d7aea0a9f1dc556f75',
    'results/c5_first_terminal_20260905_v1/layer75.pickle':'d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(EXP/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('independent archive drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text())
    hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c27_owned_journal_worker_v1.py','c27_reversible_lineage_v1.py',
        'c27_source_image_v1.py','c27_reference_transfer_v1.py','test_c27_reversible_lineage_v1.py',
        'C27_OWNED_JOURNAL_PREREG_20260911.md','C26_OWNED_EMISSION_HANDOFF_20260911.md',
        'CHECKPOINT_C26_TRANSPLANT_20260911_SHA256SUMS',*ANCHORS]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c27_reversible_lineage_v1.py',*prior['tests']]
    provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(EXP/'c27_owned_journal_worker_v1.py')]
    DIRECTORY.mkdir()
    _atomic_exclusive_json(DIRECTORY/'preregistered.json',{'source_sha256':hashes,'tests':tests,
        'provenance':provenance,'command':command,'wall_cap_s':240,'test_wall_cap_s':60,'memory_gb':16,
        'scalar_update_subtotal_whole_cap':256_000_000,'scalar_update_subtotal_branch_cap':200_000_000,
        'independent_diagnostic_work_cap':256_000_000,'construction_cap_bytes':1024**3,'entry_cap':64_000_000,
        'fresh_original_generation_for_owned_journal_only':True,'new_native_consumer_authorized':False,
        'solver_authorized':False,'old_archive_writes_authorized':False,
        'whole_integration_work_claim_authorized':False,'whole_live_path_claim_authorized':False,'formal_gain':0})
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
