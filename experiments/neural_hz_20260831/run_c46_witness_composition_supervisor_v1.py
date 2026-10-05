"""Freeze inherited evidence and the exact witness component; synthetic only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256,_atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c46_witness_composition_20260911_v1'
PRIOR=EXP/'results/c45_query_payment_20260911_v1'
ANCHORS={'result.json':'b07ce2db378fbbed325fa8af2199551ad76ce5c5af14dc6c6c64adf0fec58711',
    'exit.json':'2a44471fc22b3b71a480afc86f750353a977f70d9c08685e3fd7215723e90068'}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('prior C45 evidence drift')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('inherited source drift')
    names=[Path(__file__).name,'c46_half_lineage_extension_v1.py','c46_ordered_fixture_v1.py',
        'test_c46_half_lineage_extension_v1.py','c46_witness_composition_worker_v1.py',
        'C46_WITNESS_PREREG_20260911.md','C46_WITNESS_COMPOSITION_THEOREM_20260911.md',
        'C45_QUERY_PAYMENT_AUDIT_20260911.md','C45_SOURCE_INTEGRATION_HANDOFF_20260911.md',
        *(str(PRIOR/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c46_half_lineage_extension_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,str(EXP/'c46_witness_composition_worker_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,
        tests=tests,provenance=provenance,command=command,worker_wall_cap_s=240,test_wall_cap_s=60,
        address_space_bytes=16*1024**3,measured_transient_cap_bytes=1024**3,entry_cap=64_000_000,
        whole_diagnostic_work_cap=256_000_000,branch_work_cap=200_000_000,
        actual_source_or_native_or_solver_authorized=False,old_archive_writes_authorized=False,formal_gain=0))
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
        record.update(wall_s=time.monotonic()-started,source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),
            provenance_drift=_provenance(ROOT)!=provenance,
            old_archives_unchanged=all(_sha256(PRIOR/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
