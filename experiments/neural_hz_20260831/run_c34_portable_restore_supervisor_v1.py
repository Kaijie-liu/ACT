"""Only restore the completed archive's proved view sharing; NEVER another solve."""
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
SOURCE=EXP/'results/c34_changed_terminal_20260911_v1'
DIRECTORY=EXP/'results/c34_portable_rhs_restore_20260911_v1'
ANCHORS={
    'native_state.pickle':'cb5170da0473c01f5304ed307105df0f4aa09c4115af330126fd062ee7bbce3a',
    'terminal_gate.json':'f2b0729982efd1af933762b00eae4aee79b758b77353a8f4bf22bd30b0382906',
    'final_proof.json':'1bbd3d2326df41d0bdb7ae37f1ade1ff2996aee2e476eb1fea8d3ae42b9ede3d',
    'restore_guard.json':'f11f5bbb5b5bf65d807be366fdb9962473e0bf9e716e7e89728fe5cc3edaafb3',
    'outcome.json':'976db9fe4e9f84623dbfeb5f121487bcc124d20dd84d36b04a837312efa7d88a',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(SOURCE/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('original failed run/complete archive drift')
    prior=json.loads((SOURCE/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('entire original source drift')
    names=[Path(__file__).name,'c34_portable_rhs_restore_v1.py','c34_portable_restore_guard_v1.py',
        'test_c34_portable_rhs_restore_v1.py','C34_PORTABLE_RHS_RESTORE_PREREG_20260911.md',
        *(str(SOURCE/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    provenance=_provenance(ROOT);tests=['test_c34_portable_rhs_restore_v1.py',*prior['tests']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(EXP/'c34_portable_restore_guard_v1.py'),str(DIRECTORY)]
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',dict(source_sha256=hashes,tests=tests,
        provenance=provenance,command=command,wall_cap_s=60,test_wall_cap_s=60,memory_gb=16,
        diagnostic_work_cap=256_000_000,archive_or_production_writes_authorized=False,
        generator_native_or_solver_execution_authorized=False,decoded_RHS_identity_restoration_only=True,formal_gain=0))
    started=time.monotonic();record=dict(formal_gain=0)
    try:
        with (DIRECTORY/'tests.log').open('x') as f:
            r=subprocess.run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
                *(str(EXP/n) for n in tests)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['tests_exit_code']=r.returncode
        if r.returncode:return
        with (DIRECTORY/'worker.log').open('x') as f:
            r=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        record['worker_exit_code']=r.returncode
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance,
            old_archives_unchanged=all(_sha256(SOURCE/n)==sha for n,sha in ANCHORS.items()),
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
