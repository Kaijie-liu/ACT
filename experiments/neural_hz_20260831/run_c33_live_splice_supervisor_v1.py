"""New frozen logging correction; same unexecuted native target, no cap changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json,_sha256

EXP=Path(__file__).resolve().parent
PRIOR=EXP/'results/c32_live_splice_20260911_v1'
DIRECTORY=EXP/'results/c33_live_splice_20260911_v1'
ANCHORS={
    'closed_proof.json':'cad0401a22b403114db73d32cae4ca949d7d8314106bfc39eaccdc71345f93d5',
    'transfer_proof.json':'bab5946186e350159087a9a5d512d8e01759591e0d4309d9d2998e2936b87e86',
    'transfer_result.json':'dc75efa169d6c89c9f3346ffdb425f7f51af831de9484df3f172e6e13d4cc7c3',
    'qualification.json':'92d0bfe81173ac658b3bda4008678505ae750a84f9889d28dae7d4f6793af58a',
    'result.json':'b7dab64c1f1ec0350985705ad90b08b24e7ffdbe4c374240a3edebd717882923',
}


def main():
    if DIRECTORY.exists():raise FileExistsError(DIRECTORY)
    if any(_sha256(PRIOR/n)!=sha for n,sha in ANCHORS.items()):raise ValueError('prior failure/proof anchor drift')
    qualification=json.loads((PRIOR/'qualification.json').read_text())
    result=json.loads((PRIOR/'result.json').read_text())
    if qualification['islands'] or qualification['passed'] or result['error']!="TypeError: dict() got multiple values for keyword argument 'elapsed_s'":
        raise ValueError('not the diagnosed pre-selection event collision')
    prior=json.loads((PRIOR/'preregistered.json').read_text());hashes=dict(prior['source_sha256'])
    if any(_sha256(EXP/n)!=sha for n,sha in hashes.items()):raise ValueError('entire inherited frozen source drift')
    names=[Path(__file__).name,'c33_collision_safe_events_v1.py','c33_live_splice_worker_v1.py',
        'c33_archive_restore_guard_v1.py','test_c33_collision_safe_events_v1.py','C33_EVENT_FIX_NATIVE_PREREG_20260911.md',
        'C32_LIVE_SPLICE_AUDIT_20260911.md','CHECKPOINT_C32_LIVE_SPLICE_FAILED_20260911_SHA256SUMS',
        *(str(PRIOR/n) for n in ANCHORS)]
    hashes.update({n:_sha256(EXP/n) for n in names})
    tests=['test_c33_collision_safe_events_v1.py',*prior['tests']];provenance=_provenance(ROOT)
    command=list(prior['command']);command[:3]=[sys.executable,str(EXP/'c33_live_splice_worker_v1.py'),str(DIRECTORY)]
    command[command.index('--output')+1]=str(DIRECTORY/'result.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    freeze=dict(prior,source_sha256=hashes,tests=tests,provenance=provenance,command=command,
        change='event serializer only; exact complete worker diff regression',
        old_failed_attempt_preserved=True,original_selected_generator_was_not_executed=True,
        offline_numeric_proof_reexecution_authorized=False,completed_text_transfer_sha256=ANCHORS['transfer_proof.json'])
    DIRECTORY.mkdir();_atomic_exclusive_json(DIRECTORY/'preregistered.json',freeze)
    started=time.monotonic();record=dict(formal_gain=0,reused_completed_text_transfer_only=True)
    def run(args,log,cap):
        with (DIRECTORY/log).open('x') as stream:
            return subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=cap).returncode
    try:
        for name in ('closed_proof.json','transfer_proof.json'):
            with (DIRECTORY/name).open('xb') as f:f.write((PRIOR/name).read_bytes());f.flush();os.fsync(f.fileno())
        _atomic_exclusive_json(DIRECTORY/'live_inputs.json',dict(source_proof_sha256=ANCHORS['closed_proof.json'],
            transfer_proof_sha256=ANCHORS['transfer_proof.json'],transfer_result_sha256=ANCHORS['transfer_result.json']))
        record['tests_exit_code']=run([sys.executable,'-m','pytest','-q','--tb=short','-p','no:cacheprovider',
            *(str(EXP/n) for n in tests)],'tests.log',60)
        if record['tests_exit_code']:return
        record['worker_exit_code']=run(command,'worker.log',240)
        if record['worker_exit_code']:return
        record['restore_exit_code']=run([sys.executable,str(EXP/'c33_archive_restore_guard_v1.py'),str(DIRECTORY)],'restore.log',60)
    except subprocess.TimeoutExpired as exc:record['timeout_s']=exc.timeout
    except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        record.update(wall_s=time.monotonic()-started,
            source_drift=any(_sha256(EXP/n)!=sha for n,sha in hashes.items()),provenance_drift=_provenance(ROOT)!=provenance,
            artifacts={p.name:_sha256(p) for p in DIRECTORY.iterdir() if p.is_file()})
        _atomic_exclusive_json(DIRECTORY/'exit.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
